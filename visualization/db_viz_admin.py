from itertools import product
from sqlalchemy import create_engine
from pandas import read_sql, DataFrame
from os.path import dirname, abspath
from sys import path as sys_path
from json import dumps

try:
    sys_path.append(dirname(dirname(abspath(__file__))))
except NameError:
    print 'Working in dev mode.'

from bokeh.models import Plot, ColumnDataSource, GlyphRenderer, CustomJS
from bokeh.models.widgets.panels import Tabs, Panel
from bokeh.models.ranges import Range1d
from bokeh.models.glyphs import Line, MultiLine
from bokeh.models.tools import HoverTool, ResetTool, BoxZoomTool, PanTool, TapTool
from bokeh.layouts import Column, Row
from bokeh.models.widgets import CheckboxGroup, Button, RadioButtonGroup
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.plotting import curdoc, show
from collections import OrderedDict

from databasing.conn_strings import db_conn_string
from databasing.database_pull import pull_data_by_lift
from processing.util import process_data

# Header fields to print on top of plot
HEADER_PRINT = ['athlete_id', 'calc_reps', 'created_at', 'final_num_reps', 'lift_id', 'lift_type',
                'test_lift', 'user_comment']

# ### Effectively a global value - all possible signals that could be passed in ###
#: By default, all signals must be plotted because I only want to update the ColumnDataSource, not
#: rebuild the plot N times. To accommodate this, I will just update the alpha of the lines.
#: Start by defaulting all lines to alpha = 0
ALL_DIMS = ['_x', '_y', '_z']
# all_dims = ['_x', '_rms']
ALL_COLS = ['a', 'v', 'pwr', 'pos']
ALL_FILTS = ['_hp', '']

PLOT_Y_MIN = -0.75
PLOT_Y_MAX = 0.75

storage = dict()


class LiftPlot(object):

    plot_width = 900
    plot_height = 500

    def __init__(self, connection_string, verbose=True):

        self.connection_string = connection_string
        self.verbose = verbose

        self.plot_source = ColumnDataSource()
        self.rep_start_source = ColumnDataSource()
        self.rep_stop_source = ColumnDataSource()

        self.raw_dims = list()

        all_suffix = [x+y for (x, y) in product(ALL_DIMS, ALL_FILTS)]
        all_opts = [x+y for (x, y) in product(ALL_COLS, all_suffix)]

        self.all_cols = ALL_COLS
        self.all_dims = ALL_DIMS
        self.all_signals = all_opts

        # ###
        # TODO: Consider rearranging order of element creation and layout of app

        self._establish_inputs()

        self._establish_outputs()  # in case I want to add more later; make generalized function for it

        self.rms_plot = Plot()
        self.raw_plot = Plot()

        # Fill the layout with content (e.g. data)
        self._load_content()

        # Show/Hide signals
        self._on_signal_change(None, None, None)

        # Specifies what goes where in the app
        self._create_layout()

    def _establish_inputs(self):
        # Elements of the bokeh app that will have to be laid out later
        lift_options = [str(x) for x in sorted(self.get_data('lift_options')['lift_id'].unique())]

        #: TODO Move these inputs out of the __init__ method
        self.lift_select = Select(name='lift_id',
                                  title='Select lift_id',
                                  width=100,
                                  options=lift_options,
                                  value=lift_options[0])
        # Explicitly set callback after creation - can't do it in creation step??
        self.lift_select.on_change('value', self._on_lift_change)

        self.signal_select = CheckboxGroup(
            name='signal_select',
            width=80,
            height=150,
            labels=self.all_cols,
            active=[0]
        )
        self.signal_select.on_change('active', self._on_signal_change)

        self.dim_select = CheckboxGroup(
            name='dimension_select',
            width=80,
            height=150,
            labels=self.all_dims,
            # active=range(len(self.all_dims))
            active=[2]
        )
        self.dim_select.on_change('active', self._on_signal_change)

        #: Careful with this - button to delete all data associated with current lift
        #: For use in post-hoc data cleaning.
        self.del_button = Button(label='DELETE LIFT', button_type='danger', width=100, height=30)
        self.del_button.on_click(self._del_click)
        self.del_button_text = 'N/A'

        #: Button to delete all rep start/stop points in given lift
        self.del_rep_button = Button(label='DELETE REPS', button_type='danger', width=100, height=30)
        self.del_rep_button.on_click(self._del_reps)

        #: To switch TapTool action to assign either "rep_start" or "rep_stop" on user tap
        self.tap_select = RadioButtonGroup(
            name='rep_tool_event_type',
            width=75,
            height=100,
            labels=['rep', 'eccentric', 'concentric'],
            active=0
        )
        self.tap_select.on_click(self._event_type_change)

        self.tap_action = RadioButtonGroup(
            name='rep_tool_action',
            width=75,
            height=100,
            labels=['_start', '_stop', '_delete'],
            active=0
        )
        #: Used to fill the text box next to the tap_select
        #: Will update with taptool callbacks
        self.rep_info_text = ''

        self.calc_button = Button(label='Calculate', button_type='primary', width=100, height=30)
        self.calc_button.on_click(self._run_per_rep_calc)

    def _establish_outputs(self):
        # Has to be initialized before I can set the text.
        self.lift_info = Div(width=500, height=60)
        # To print success/fail
        self.del_button_info = Div(width=200, height=20)
        self.rep_info = Div(width=100, height=100)
        self.calc_button_info = Div(width=300, height=100, text='Ready')

    def _create_layout(self):

        #: del_header contains the delete button and a text field
        self.del_header = Column(width=150, height=80)
        self.del_header.children = [self.del_button, self.del_rep_button, self.del_button_info]

        #: calc_header contains calc_button and a text field for the outputs
        self.calc_header = Column(width=300, height=100)
        self.calc_header.children = [self.calc_button, self.calc_button_info]

        #: Format the row that the tap_select tool will be in
        self.h_filler = Div(width=5, height=100)
        self.tap_select_row = Row(width=600, height=100)
        self.tap_select_row.children = [self.h_filler, self.tap_select, self.tap_action, self.rep_info, self.calc_header]

        #: right_header contains the text box with lift metadata and the tap_select buttongroup
        self.right_header = Column(width=500, height=150)
        self.right_header.children = [self.lift_info, self.tap_select_row]

        #: plot_header contains all input tools, text boxes, etc that sit above the plot
        self.plot_header = Row(width=self.plot_width+100, height=185)
        self.plot_header.children = [self.lift_select, self.signal_select, self.dim_select, self.right_header, self.del_header]

        # ## RMS PLOT ##

        # Box that contains the RMS plot (Box may be unnecessary)
        self.rms_panel_box = Column(width=self.plot_width, height=self.plot_height)
        self.rms_panel_box.children = [self.rms_plot]

        # Panel that contains the RMS box
        self.panel_rms = Panel(
            child=self.rms_plot, title='RMS Plot', closable=False, width=self.plot_width, height=self.plot_height)

        # ## RAW PLOT ##

        self.raw_panel_box = Column(width=self.plot_width, height=self.plot_height)
        self.raw_panel_box.children = [self.raw_plot]

        # self.panel_raw = Column(
        #     child=self.raw_plot, title='Raw Plot', closable=False, width=self.plot_width, height=self.plot_height)
        self.panel_raw = Column(children=[self.raw_plot], width=self.plot_width, height=self.plot_height)

        # ##

        # Contains ALL panels
        # self.panel_parent = Tabs(width=self.plot_width+10, height=self.plot_height, active=0)
        # self.panel_parent.tabs = [self.panel_raw]
        self.panel_parent = Column(width=self.plot_width+10, height=self.plot_height)
        self.panel_parent.children = [self.panel_raw]

        self.layout = Column(children=[self.plot_header, self.panel_raw], width=self.plot_width+20, height=self.plot_height)

    def _load_content(self):
        self.update_datasource()
        self.raw_plot = self.make_raw_plot(self.plot_source, self.rep_start_source, self.rep_stop_source)

    #: Controls behavior of Checkboxgroup Selection tool
    def _on_signal_change(self, attr, old, new):
        N = len(self.all_dims)  # number of dimensions per signal (x, y, z)
        for i in range(len(self.signal_select.labels)):
            is_active = i in self.signal_select.active  # whether this label (e.g. 'a', 'v' is active)
            for ix in range(i*N, (i+1)*N):  # apply to all dims associated with this label
                self.raw_plot.renderers[ix].visible = is_active & ((ix-i*N) in self.dim_select.active)
                # if self.raw_plot.renderers[ix].visible:
                #     print 'plotting {}'.format(self.raw_plot.renderers[ix].name)

    #: Controls behavior of dropdown Select tool
    def _on_lift_change(self, attr, old, new):
        print 'Updating plot with lift_id: {}'.format(new)
        self._reset_tap_buttons()
        self.update_datasource()

    def _reset_tap_buttons(self):
        self.tap_action.active = 0
        self.tap_select.active = 0

    def _del_reps(self):
        lift_id = self.lift_select.value
        print 'Deleting rep start/stop points for lift_id {}...'.format(lift_id)

        sql = '''
        DELETE FROM lift_event WHERE lift_id = {lid};
        '''.format(lid=lift_id)

        conn = create_engine(self.connection_string)
        ret = conn.execute(sql)
        ret.close()

        text = 'Deleted rep start/stop points for lift_id {} from table "lift_event"'.format(lift_id)

        self.del_button_text = text
        self.rep_info_text = ''  # clear rep_info_text, if not already cleared

        # refresh the plot
        self.update_datasource()

    def _del_click(self):
        lift_id = self.lift_select.value
        print 'Deleting data for lift_id {}...'.format(lift_id)

        sql = '''
        DELETE FROM lifts WHERE lift_id = {id};
        DELETE FROM lift_data WHERE lift_id = {id};
        '''.format(id=lift_id)

        conn = create_engine(self.connection_string)
        ret = conn.execute(sql)
        ret.close()

        text = 'Executed deletion for lift_id {id} from "tables" lifts and "lift_data"'.format(id=lift_id)

        print text
        self.del_button_text = text

        # remove cached data, if exists
        storage.pop((lift_id, 'header'), None)  # default to None, i.e. do nothing, if key does not exist
        storage.pop((lift_id, 'data'), None)

        # remove lift_id from lift_select options
        self.lift_select.options.remove(str(lift_id))

        # change active lift_id to a default, which should trigger _on_lift_change and
        # cascade all proper function calls
        self.lift_select.value = self.lift_select.options[0]

    def _get_e_type(self):
        return self.tap_select.labels[self.tap_select.active]

    def _run_per_rep_calc(self):
        lift_id = int(self.lift_select.value)
        print 'Running rep calculations for all reps in lift {}...'.format(lift_id)

        #: Retrieve all rep events for this lift
        sql = '''
        SELECT * FROM lift_event WHERE lift_id = {};
        '''.format(lift_id)

        conn = create_engine(self.connection_string)
        events = read_sql(sql, conn)

        #: Filter out anything that is not a start/stop point of the appropriate event type
        e_type = self._get_e_type()
        events = events.loc[[e_type in x for x in events['event']]].sort_values(by='timepoint', ascending=True)

        if events.shape[0] > 0:
            #: Group the start and stop points
            #: NOTE: assumes the structure of df events is alternating start/stop
            #        if this is not true, this logic will probably break!
            rep = 0
            events['rep_num'] = 0
            for i, row in events.iterrows():
                if (i % 2 == 0) & (row['event'] == 'rep_start'):
                    events.loc[i, 'rep_num'] = rep
                elif (i % 2 == 1) & (row['event'] == 'rep_stop'):
                    events.loc[i, 'rep_num'] = rep
                else:
                    print 'Unhandled event, row number {i} shows event {e}. Fix this.'.format(i=i, e=row['event'])

                #: Increase rep number every 2 iterations, which should capture a full start/stop pair
                if (i > 0) & ((i % 2) == 1):
                    rep += 1

            #: pair each rep's start and stop points
            ts = events.groupby('rep_num').apply(lambda df: list(df['timepoint']))
            ts.name = 't_pair'

            #: run calculations on source data between timepoints
            data = storage[(lift_id, 'data')]

            # TODO: Try to clean this up
            rep_info = DataFrame(columns=['v_max', 'v_mean', 'pwr_max', 'pwr_mean', 'N'], index=ts.index)
            for i, row in ts.iteritems():
                dat = data.loc[(data['timepoint'] >= row[0]) & (data['timepoint'] <= row[1]) ]
                rep_info.loc[i, 'v_mean'] = round(dat['v_rms_raw_hp'].mean(), 2)
                rep_info.loc[i, 'v_max'] = round(dat['v_rms_raw_hp'].max(), 2)
                rep_info.loc[i, 'pwr_mean'] = round(dat['pwr_rms_raw_hp'].mean(), 2)
                rep_info.loc[i, 'pwr_max'] = round(dat['pwr_rms_raw_hp'].max(), 2)
                rep_info.loc[i, 'N'] = dat.shape[0]

            #: Run calculations over all timepoints for total average
            rep_info.reset_index(inplace=True)
            rep_info['rep_num'] = (rep_info['rep_num'] + 1).astype(str)
            rep_info = rep_info.append(DataFrame(
                data={'rep_num': 'Overall',
                      'v_max': rep_info['v_max'].max(),
                      'v_mean': (rep_info['v_mean']*rep_info['N']).sum()/float(rep_info['N'].sum()),
                      'pwr_max': rep_info['pwr_max'].max(),
                      'pwr_mean': (rep_info['pwr_mean'] * rep_info['N']).sum() / float(rep_info['N'].sum())
                      }, index=[0]
                ), ignore_index=True)

            #: Don't need column N any more
            rep_info.drop('N', axis=1, inplace=True)
            rep_info.set_index('rep_num', inplace=True)

            text = rep_info.reset_index().to_string(index=False).replace('\n', '<br>').replace('    ', '&emsp;').replace('  ', '&emsp;')
            self.calc_button_info.text = text
        else:
            text = 'Couldnt find start/stop pairs of type {}.\nCannot calculate anything'.format(e_type)
            self.calc_button_info.text = text

        print 'done with calculations'

    def update_datasource(self):

        # ### Prep the data ###

        # ## A/V/P plot data ##

        #: In case this was triggered by the delete button
        self.del_button_info.text = self.del_button_text

        header, data = self.get_data('lift_data')

        # Lift metadata to display
        header['created_at'] = header['created_at'].strftime('%Y-%m-%d')
        header['ended_at'] = header['ended_at'].strftime('%Y-%m-%d')
        header['updated_at'] = header['updated_at'].strftime('%Y-%m-%d')

        # It's unknown ahead of time which raw dimensions will be present in each lift, so we need to find
        # them all dynamically. We known RMS will be present for a, v, p.
        raw_headers = [x.split('_')[-1] for x in data.columns if ('rms' not in x) and ('time' not in x)]
        self.raw_dims = list(set(raw_headers))  # remove duplicates

        data['zero'] = 0.
        data['x_axis'] = data['timepoint']/max(data['timepoint'])  # scale x axis to be (0, 1)

        # ## Rep start/stop lines from database ##
        self.rep_info.text = self.rep_info_text

        rep_dat = self.get_data('rep_data')

        #: decide which start/stop points to plot
        e_type = self._get_e_type()

        if rep_dat is not None:
            # start_dat, stop_dat = self.format_lift_event_data(
            #     rep_dat, e_type, min(data['a_x']), max(data['a_x']), max(data['timepoint']))
            start_dat, stop_dat = self.format_lift_event_data(
                rep_dat, e_type, PLOT_Y_MIN, PLOT_Y_MAX, max(data['timepoint'])
            )
            start_src = ColumnDataSource(start_dat)
            stop_src = ColumnDataSource(stop_dat)
        else:
            #: Empty column data source for lines to be drawn onto plot
            start_src = ColumnDataSource({'xs': [], 'ys': []})
            stop_src = ColumnDataSource({'xs': [], 'ys': []})

        # update number of starts and stops present
        # TODO - figure out where to put these??
        n_starts = len(start_src.data['xs'])
        n_stops = len(stop_src.data['xs'])

        # subset header data
        print_header = OrderedDict([(k, header[k]) for k in HEADER_PRINT])
        print_header.update({'starts': n_starts, 'stops': n_stops})
        self.lift_info.text = dumps(print_header)

        # ### Update the data sources ###

        #: The plot is set up to accept all values (e.g. a_x, a_x_rms, v_x, p_x_rms, etc)
        self.plot_source.data = ColumnDataSource(data).data
        self.plot_source.column_names = self.plot_source.data.keys()

        #: Update the rep start/stop data sources dynamically
        self.rep_start_source.data = start_src.data
        self.rep_start_source.column_names = self.rep_start_source.data.keys()

        self.rep_stop_source.data = stop_src.data
        self.rep_stop_source.column_names = self.rep_stop_source.data.keys()

        # print 'done updating plot datasource'

    def make_raw_plot(self, source, rep_start_source, rep_stop_source):
        tooltips = '''<div><span style="font-size: 12px;"> <b>time  :</b> @timepoint s</span></div>
                      <div><span style="font-size: 12px;"> <b>acc   :</b> @a_x m/s^2</span></div>
                      <div><span style="font-size: 12px;"> <b>acc HP:</b> @a_x_hp m/s^2</span></div>
                      <div><span style="font-size: 12px;"> <b>vel HP:</b> @v_x_hp m/s</span></div>
                      <div><span style="font-size: 12px;"> <b>pwr HP:</b> @pwr_x_hp W</span></div>
                      <div><span style="font-size: 12px;"> <b>pos HP:</b> @pos_x_hp m</span></div>'''

        #: Gold line that moves to where user clicks to show which timepoint will be logged on use of TapTool
        src = ColumnDataSource(
                dict(
                    x=[0., 0.],
                    # y=[min(source.data['a_x']), max(source.data['a_x'])]
                    y=[PLOT_Y_MIN, PLOT_Y_MAX]
                )
            )

        draw_line_cb = CustomJS(
            args=dict(src=src, source=source),
            code="""
                // 0d level specific to taptool - only returns one value, the index of the value tapped
                // Use this index "idx" to extract the timepoint
                var idx = cb_obj.selected['0d'].indices[0];

                // Use idx to extract from timepoint array ts
                var ts = source.data['timepoint'];
                var t = ts[idx];
                console.log("Registered hit at timepoint: " + t);

                // timepoint is raw - need to scale to match x-axis scale of [0, 1]
                // use datasource used to construct lines in plot to provide timepoint array
                var t_scale = t/ts[ts.length-1];

                // update data source
                src.data['x'] = [t_scale, t_scale];

                // Upper/Lower bounds of plot determined by a_x signal
                // data['y'] = [min(data['a_x']), max(data['a_x']);
                // DON'T WANT TO UPDATE y-vals

                // print data source "src" with updated data
                // console.log(src.data);

                // trigger update of data source "src"
                src.trigger('change');
                """
            )

        plot = Plot(
            title=None,
            x_range=Range1d(min(source.data['x_axis']), max(source.data['x_axis'])),
            y_range=Range1d(PLOT_Y_MIN, PLOT_Y_MAX),  # all signals have been max-min normalized
            plot_width=self.plot_width,
            plot_height=self.plot_height,
            h_symmetry=False,
            v_symmetry=False,
            min_border=0,
            toolbar_location='right',
            logo=None
        )

        rends = list()
        for sig in ['a_', 'v_', 'pwr_', 'pos_']:
            for dim in ['x_hp', 'y_hp', 'z_hp']:
                y_val = sig + dim

                #: Split out signal type (a/v/p) by color
                if 'a' in y_val:
                    style = 'solid'
                elif 'v' in y_val:
                    style = 'dotted'
                elif 'pwr' in y_val:
                    style = 'dashed'
                elif 'pos' in y_val:
                    style = 'dashdot'
                else:
                    #: Uncaught line type here - make red so we can see it easily
                    style = 'dashdot'

                #: Differentiate between high-passed signal and non-HP signal
                if dim == 'x_hp':
                    c = 'purple'
                elif dim == 'y_hp':
                    c = 'blue'
                elif dim == 'z_hp':
                    c = 'black'
                else:
                    c = 'red'

                l = Line(x='x_axis', y=y_val, name=y_val, line_color=c, line_dash=style, line_alpha=1)
                rend = GlyphRenderer(data_source=source, glyph=l, name=y_val)
                rends.append(rend)

                if y_val == 'a_z_hp':
                    #: Explicitly set the a_z glyph to be the basis of the hover tool
                    h_rends = [rend]

        hover = HoverTool(renderers=h_rends,tooltips=tooltips, point_policy='follow_mouse')

        # #: Build renderer for lines generated by tap tool
        tap_line = Line(x='x', y='y', name='rep_tool', line_color='gold', line_dash='dashed', line_alpha=1., line_width=2)
        tap_line_renderer = GlyphRenderer(data_source=src, glyph=tap_line, name='tap_line')
        rends.append(tap_line_renderer)

        #: Plot responsive (e.g. updates) multiline glyphs for all start points and stop points
        rep_starts_glyph = MultiLine(xs='xs', ys='ys', line_color='green', line_dash='dashed', line_width=2)
        rep_starts_rend = GlyphRenderer(data_source=rep_start_source, glyph=rep_starts_glyph, name='rep_starts')
        rends.append(rep_starts_rend)

        rep_stops_glyph = MultiLine(xs='xs', ys='ys', line_color='red', line_dash='dashed', line_width=2)
        rep_stops_rend = GlyphRenderer(data_source=rep_stop_source, glyph=rep_stops_glyph, name='rep_stops')
        rends.append(rep_stops_rend)

        # tap_rends = [tap_line_renderer] + h_rends
        taptool = TapTool(renderers=h_rends, callback=draw_line_cb)

        #: can this be moved??
        source.on_change('selected', self.tap_callback)

        #: Build renderer for default zero line; useful as reference
        zero_line = Line(x='x_axis', y='zero', name='zero', line_color='red', line_dash='dashed', line_alpha=1)
        zero_line_renderer = GlyphRenderer(data_source=source, glyph=zero_line, name='zero_rend')

        rends.append(zero_line_renderer)

        zoom = BoxZoomTool()
        reset = ResetTool()
        pan = PanTool()
        plot.renderers.extend(rends)
        plot.tools.extend([hover, zoom, reset, pan, taptool])
        # TODO figure out legend
        # plot.legend.append([a_line])
        # plot.legend.location = 'upper_left'

        return plot

    def tap_callback(self, attr, old, new):
        # dict with 0d, 1d, 2d attributes of "selected" attribute of the callback object
        idx = new['0d']['indices'][0]

        # Use the index to find the proper timepoint
        ts = self.plot_source.data['timepoint']
        t = ts[idx]

        #: Other pieces of app metadata
        #: Parse out what label (rep, eccentric, concentric) and action (_start, _stop, _delete)
        #: to apply when tap is triggered
        label = self.tap_select.labels[self.tap_select.active]
        action = self.tap_action.labels[self.tap_action.active]
        event = label + action

        #: Find lift_id
        lift_id = int(self.lift_select.value)

        print 'lift_id {i}: Registered {e} click on timepoint {t}'.format(i=lift_id, e=event, t=t)

        if action == '_delete':
            event = self.find_nearest_event(label, t, t_lim=1.)
            if event is not None:
                t_near = event['timepoint']
                self._remove_lift_event(lift_id, t_near)
        else:
            self._add_lift_event(lift_id, event, t)

        #: Update datasources after modifying lift_event table
        self.update_datasource()

    def _event_type_change(self, new):
        self.update_datasource()

    def find_nearest_event(self, e_type, t, t_lim=1.):
        conn = create_engine(self.connection_string)

        query = '''
        SELECT * FROM lift_event
        WHERE lift_id = {id}
            AND timepoint BETWEEN {t1} AND {t2}
        '''.format(id=int(self.lift_select.value), t1=t-t_lim, t2=t+t_lim)

        #: Retrieve all lift_event items within the timeframe, if any
        events = read_sql(query, conn)

        if events.shape[0] > 0:
            #: only want to delete events of same e_type as what is active on tap_select tool
            event_mask = events['event'].apply(lambda x: e_type in x)
            filt_events = events.loc[event_mask, :].reset_index(drop=True)

            if filt_events.shape[0] > 0:
                # sort so that min(timepoint) comes first
                filt_events = filt_events.sort_values(by='timepoint', ascending=True).reset_index(drop=True)
                nearest_event = filt_events.ix[0]  # returns a pandas Series
            else:
                nearest_event = None
                text = 'No events of type {e} within {lim} seconds of tap at {t}'.format(e=e_type, lim=t_lim, t=t)
                self.rep_info_text = text
                # print 'Returning nearest event: \n{}'.format(nearest_event)
        else:
            nearest_event = None
            text = 'No events of type {e} within {lim} seconds of tap at {t}'.format(e=e_type, lim=t_lim, t=t)
            self.rep_info_text = text

        return nearest_event

    def _add_lift_event(self, lift_id, event, timepoint):
        conn = create_engine(self.connection_string)

        query = '''
        INSERT INTO lift_event (lift_id, timepoint, event) VALUES
        ({id}, {t}, '{e}')
        '''.format(id=lift_id, t=timepoint, e=event)

        text = '--- Executed --- \n{}'.format(query)

        self.rep_info_text = text

        conn.execute(query)

    def _remove_lift_event(self, lift_id, timepoint):
        conn = create_engine(self.connection_string)

        query = '''
        DELETE FROM lift_event
        WHERE lift_id = {id}
            AND timepoint = {t}
        '''.format(id=lift_id, t=timepoint)

        text = '--- Executed --- \n{}'.format(query)

        self.rep_info_text = text

        conn.execute(query)

    def _proc_non_rms(self, signal, label, hp, sigtype):
        # for col in accel.columns:
        #: Because I changed process_data to return a Series regardless of whether or not RMS is True,
        #: some logic downstream has been impacted and needed to be updated.
        col = sigtype
        raw_col = str(col) + label
        signal[raw_col] = signal[col]
        signal[col + label] = self.max_min_scale(signal[col])
        #: If highpass, a_x will be present (cause the column won't be overwritten; a new column is
        #: created), but we don't want to keep it.
        if hp:
            signal = signal.drop(col, axis=1)

        return signal

    def get_data(self, set_name):

        conn = create_engine(self.connection_string)

        if set_name == 'lift_options':
            query = '''
            SELECT
                al.lift_id,
                al.lift_type,
                ai.athlete_last_name || ', ' || ai.athlete_first_name AS athlete_name,
                ai.athlete_id
            FROM lifts AS al
            INNER JOIN athletes AS ai
                ON al.athlete_id = ai.athlete_id
            ORDER BY al.lift_type, al.lift_id DESC
            '''
            values = read_sql(query, conn)

            return values
        elif set_name == 'users':
            print 'Not implemented.'
            return None
        elif set_name == 'lift_data':
            if (int(self.lift_select.value), 'data') in storage.keys():
                # print 'Key ({}, {}) found in storage dict'.format(int(self.lift_select.value), 'data')
                header = storage[(int(self.lift_select.value), 'header')]
                data = storage[(int(self.lift_select.value), 'data')]
                return header.copy(), data.copy()
            else:
                # print 'Key ({}, {}) NOT found in storage dict'.format(int(self.lift_select.value), 'data')
                header, data = pull_data_by_lift(int(self.lift_select.value))

                for hp, lab in [(True, '_hp'), (False, '')]:
                    a_rms, v_rms, pwr_rms, pos_rms, force_rms = process_data(header, data, RMS=True, highpass=hp)

                    accel, vel, pwr, pos, force = process_data(header, data, RMS=False, highpass=hp)
                    gyro = accel[['g_x', 'g_y', 'g_z']]
                    accel = accel.drop(['timepoint', 'lift_id'] + list(gyro.columns), axis=1)
                    if 'millis' in accel.columns:
                        accel.drop('millis', axis=1, inplace=True)

                    #: If first loop, instantiate empty dataframe dat with proper index
                    if hp:
                        dat = DataFrame(index=a_rms.index)

                    for dim in ['x', 'y', 'z']:
                        accel = self._proc_non_rms(accel, lab, hp, sigtype='a_'+dim)
                        vel = self._proc_non_rms(vel, lab, hp, sigtype='v_'+dim)
                        pwr = self._proc_non_rms(pwr, lab, hp, sigtype='pwr_'+dim)
                        pos = self._proc_non_rms(pos, lab, hp, sigtype='pos_'+dim)
                        force = self._proc_non_rms(force, lab, hp, sigtype='force_'+dim)

                    #: On first loop, dat should be empty dataframe with overlapping indices,
                    #: so these joins should be fine. On second loop, dat will already have half the data.
                    dat = dat.join(DataFrame(
                        data={
                            'a_rms'+lab: self.max_min_scale(a_rms['rms']),
                            'a_rms_raw'+lab: a_rms,
                            'v_rms'+lab: self.max_min_scale(v_rms['rms']),
                            'v_rms_raw'+lab: v_rms,
                            'pwr_rms'+lab: self.max_min_scale(pwr_rms['rms']),
                            'pwr_rms_raw'+lab: pwr_rms,
                            'pos_rms'+lab: self.max_min_scale(pos_rms['rms']),
                            'pos_rms_raw'+lab: pos_rms,
                            'force_rms'+lab: self.max_min_scale(force_rms['rms']),
                            'force_rms_raw'+lab: force_rms
                            },
                        index=a_rms.index).join(accel).join(vel).join(pwr).join(pos).join(force))

                    #: Only want timepoint series once, so save until the end
                    if not hp:
                        dat = dat.join(data['timepoint'])

                storage[(int(self.lift_select.value), 'data')] = dat
                storage[(int(self.lift_select.value), 'header')] = header
                return header.copy(), dat.copy()
        elif set_name == 'rep_data':
            query = '''
            SELECT
                *
            FROM lift_event
            WHERE lift_id = {}
            '''.format(self.lift_select.value)

            data = read_sql(query, conn)
            if data.shape[0] > 0:
                return data
            else:
                print 'Nothing in table lift_event for lift_id {}'.format(self.lift_select.value)
                return None

    @staticmethod
    def format_lift_event_data(df, e_type, y_min, y_max, t_max):
        #: df is dataframe with columns lift_id, timepoint, event
        #: timepoint = time at which event occurs
        #: event = e_type (e.g. rep, eccentric, concentric) + action (_start, _stop)

        #: Bokeh MultiLine wants a series of points of the format
        #: [[x11, x12], [y11, y12]], [[x21, x22], [y21, y22]], ...
        df['x_val'] = df['timepoint']/float(t_max)  # scales all timepoints to be between 0 and 1 to fit on axis
        df['xs'] = df['x_val'].apply(lambda x: [x]*2)
        ys = [y_min, y_max]
        df['ys'] = [ys]*df.shape[0]  # replicate list "ys" df.shape[0] times

        df.drop(['x_val', 'timepoint', 'lift_id'], axis=1, inplace=True)

        #: Divvy up data according to whether each row reflects a rep start or stop
        e_start = e_type + '_start'
        e_stop = e_type + '_stop'
        starts = df.loc[df['event'] == e_start].drop('event', axis=1).reset_index(drop=True)
        stops = df.loc[df['event'] == e_stop].drop('event', axis=1).reset_index(drop=True)

        return starts, stops

    @staticmethod
    def max_min_scale(x):
        # return (x - min(x))/(max(x) - min(x))
        return x/(max(x) - min(x))

app = LiftPlot(db_conn_string, verbose=True)
curdoc().add_root(app.layout)
