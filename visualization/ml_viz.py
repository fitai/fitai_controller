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
from bokeh.plotting import curdoc

from databasing.db_conn_strings import conn_string
from databasing.database_pull import pull_data_by_lift
from processing.util import process_data


storage = dict()


class LiftPlot(object):

    plot_width = 900
    plot_height = 500

    def __init__(self, connection_string, verbose=True):

        self.connection_string = connection_string
        self.verbose = verbose

        self.plot_source = ColumnDataSource()

        self.raw_dims = list()

        # ### Effectively a global value - all possible signals that could be passed in ###
        #: By default, all signals must be plotted because I only want to update the ColumnDataSource, not
        #: rebuild the plot N times. To accommodate this, I will just update the alpha of the lines.
        #: Start by defaulting all lines to alpha = 0
        # TODO Confirm that app works when all_dims actually contains all possible dimensions
        # all_dims = ['_x', '_y', '_z', '_rms']
        all_dims = ['_x', '_rms']
        all_cols = ['a', 'v', 'p']
        all_filts = ['_hp', '']

        all_suffix = [x+y for (x, y) in product(all_dims, all_filts)]
        all_opts = [x+y for (x, y) in product(all_cols, all_suffix)]

        self.all_signals = all_opts
        self.active_signals = list()  # to be filled in each time update_datasource() is called

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
            width=100,
            height=90,
            labels=['a', 'v', 'p', 'a_hp', 'v_hp', 'p_hp'],
            active=[0]
        )
        self.signal_select.on_change('active', self._on_signal_change)

        #: Careful with this - button to delete all data associated with current lift
        #: For use in post-hoc data cleaning.
        self.del_button = Button(label='DELETE LIFT', button_type='danger', width=100, height=30)
        self.del_button.on_click(self._del_click)
        self.del_button_text = 'N/A'

    def _establish_outputs(self):
        # Has to be initialized before I can set the text.
        self.lift_info = Div(width=500, height=50)
        # To print success/fail
        self.del_button_info = Div(width=200, height=20)
        self.rep_info = Div(width=350, height=100)

    def _create_layout(self):

        #: del_header contains the delete button and a text field
        self.del_header = Column(width=150, height=50)
        self.del_header.children = [self.del_button, self.del_button_info]

        #: Format the row that the tap_select tool will be in
        self.h_filler = Div(width=50, height=100)
        self.tap_select_row = Row(width=500, height=100)
        self.tap_select_row.children = [self.h_filler, self.tap_select, self.rep_info]

        #: right_header contains the text box with lift metadata and the tap_select buttongroup
        self.right_header = Column(width=500, height=150)
        self.right_header.children = [self.lift_info, self.tap_select_row]

        #: plot_header contains all input tools, text boxes, etc that sit above the plot
        self.plot_header = Row(width=self.plot_width, height=150)
        self.plot_header.children = [self.lift_select, self.signal_select, self.right_header, self.del_header]

        # ## RAW PLOT ##

        self.raw_panel_box = Column(width=self.plot_width, height=self.plot_height)
        self.raw_panel_box.children = [self.raw_plot]

        self.panel_raw = Panel(
            child=self.raw_plot, title='Raw Plot', closable=False, width=self.plot_width, height=self.plot_height)

        # ##

        # Contains ALL panels
        self.panel_parent = Tabs(width=self.plot_width+10, height=self.plot_height, active=0)
        self.panel_parent.tabs = [self.panel_rms, self.panel_raw]

        self.layout = Column(children=[self.plot_header, self.panel_parent], width=self.plot_width+20, height=self.plot_height)

    def _load_content(self):
        self.update_datasource()
        self.rms_plot = self.make_RMS_plot(self.plot_source)
        self.raw_plot = self.make_raw_plot(self.plot_source, self.rep_start_source, self.rep_stop_source)

    #: Controls behavior of Checkboxgroup Selection tool
    def _on_signal_change(self, attr, old, new):
        for i in range(len(self.signal_select.labels)):
            # print 'setting renderer {i} to {tf}'.format(
            #     i=self.signal_select.labels[i], tf=i in self.signal_select.active)

            #: If renderer i is in self.signal_select.active (list[0, 1, 2]), then set visible to true
            #: Else visible is false and signal is plotted but not shown
            # print [rend.name for rend in self.raw_plot.renderers]
            self.raw_plot.renderers[i].visible = i in self.signal_select.active
            # self.raw_plot.renderers[i+3].visible = i in self.signal_select.active

            #: For now, raw plot has more renderers in it than the RMS plot does
            if i > len(self.rms_plot.renderers)-1:
                continue

            self.rms_plot.renderers[i].visible = i in self.signal_select.active

    #: Controls behavior of dropdown Select tool
    def _on_lift_change(self, attr, old, new):
        print 'Updating plot with lift_id: {}'.format(new)
        self.update_datasource()

    def _del_click(self, *args):
        lift_id = self.lift_select.value
        print 'Deleting data for lift_id {}...'.format(lift_id)

        sql = '''
        DELETE FROM athlete_lift WHERE lift_id = {id};
        DELETE FROM lift_data WHERE lift_id = {id};
        '''.format(id=lift_id)

        conn = create_engine(self.connection_string)
        ret = conn.execute(sql)
        ret.close()

        text = 'Executed deletion for lift_id {id} from tables athlete_lift and lift_data'.format(id=lift_id)

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

    def update_datasource(self):

        # ### Prep the data ###

        # ## A/V/P plot data ##

        #: In case this was triggered by the delete button
        self.del_button_info.text = self.del_button_text

        header, data = self.get_data('lift_data')

        # Lift metadata to display
        header['lift_start'] = header['lift_start'].strftime('%Y-%m-%d')
        self.lift_info.text = dumps(header)

        # It's unknown ahead of time which raw dimensions will be present in each lift, so we need to find
        # them all dynamically. We known RMS will be present for a, v, p.
        raw_headers = [x.split('_')[-1] for x in data.columns if ('rms' not in x) and ('time' not in x)]
        self.raw_dims = list(set(raw_headers))  # remove duplicates

        data['zero'] = 0.
        data['x_axis'] = data['timepoint']/max(data['timepoint'])  # scale x axis to be (0, 1)

        # ## Rep start/stop lines from database ##
        self.rep_info.text = self.rep_info_text

        rep_dat = self.get_data('rep_data')

        if rep_dat is not None:
            start_dat, stop_dat = self.format_lift_event_data(
                rep_dat, min(data['a_x']), max(data['a_x']), max(data['timepoint']))
            start_src = ColumnDataSource(start_dat)
            stop_src = ColumnDataSource(stop_dat)
        else:
            #: Empty column data source for lines to be drawn onto plot
            start_src = ColumnDataSource({'xs': [], 'ys': []})
            stop_src = ColumnDataSource({'xs': [], 'ys': []})

        # ### Update the data sources ###

        #: The plot is set up to accept all values (e.g. a_x, a_x_rms, v_x, p_x_rms, etc)
        self.plot_source.data = ColumnDataSource(data).data
        self.plot_source.column_names = self.plot_source.data.keys()

        #: Update the rep start/stop data sources dynamically
        self.rep_start_source.data = start_src.data
        self.rep_start_source.column_names = self.rep_start_source.data.keys()

        self.rep_stop_source.data = stop_src.data
        self.rep_stop_source.column_names = self.rep_stop_source.data.keys()

        print 'done updating plot datasource'

    def make_RMS_plot(self, source):
        tooltips = '''<div><span style="font-size: 12px;"> <b>time:</b> @timepoint s</span></div>
                      <div><span style="font-size: 12px;"> <b>accel:</b> @a_rms_raw m/s^2</span></div>
                      <div><span style="font-size: 12px;"> <b>vel:</b> @v_rms_raw m/s</span></div>
                      <div><span style="font-size: 12px;"> <b>pwr:</b> @p_rms_raw W</span></div>'''

        plot = Plot(
            title=None,
            x_range=Range1d(min(source.data['x_axis']), max(source.data['x_axis'])),
            # y_range=Range1d(min(self.plot_source.data['a_rms']), max(self.plot_source.data['a_rms'])),
            y_range=Range1d(0, 1),
            plot_width=self.plot_width,
            plot_height=self.plot_height,
            h_symmetry=False,
            v_symmetry=False,
            min_border=0,
            toolbar_location='right',
            logo=None
        )

        rends = list()
        # for y_val in [x for x in self.all_signals if 'rms' in x]:
        for y_val in ['a_rms', 'v_rms', 'p_rms']:

            if 'a' in y_val:
                c = 'black'
            elif 'v' in y_val:
                c = 'blue'
            elif 'p' in y_val:
                c = 'purple'
            else:
                c = 'red'

            l = Line(x='x_axis', y=y_val, name=y_val, line_color=c, line_alpha=1)
            rend = GlyphRenderer(data_source=source, glyph=l, name=y_val)
            rends.append(rend)

            if y_val == 'a_rms':
                h_rends = [rend]

        # axis_theme = dict(
        #     axis_label=None, axis_label_text_font_size='0pt', minor_tick_line_alpha=0.0,
        #     axis_line_alpha=0.0, major_tick_line_alpha=0.0, major_label_text_color='grey',
        #     major_label_text_font_size='0pt')
        #
        # plot.add_layout(LinearAxis(**axis_theme), 'left')
        # plot.add_layout(LinearAxis(**axis_theme), 'below')

        hover = HoverTool(renderers=h_rends, tooltips=tooltips, point_policy='follow_mouse')
        zoom = BoxZoomTool()
        reset = ResetTool()
        pan = PanTool()
        plot.renderers.extend(rends)
        plot.tools.extend([zoom, reset, pan, hover])
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
        #: Parse out whether rep_start or rep_stop
        label = self.tap_select.labels[self.tap_select.active]
        #: Find lift_id
        lift_id = int(self.lift_select.value)

        print 'lift_id {i}: Registered {l} click on timepoint {t}'.format(i=lift_id, l=label, t=t)

        if label == 'delete_nearby':
            event = self.find_nearest_event(t, t_lim=1.)
            if event is not None:
                t_near = event['timepoint']
                self._remove_lift_event(lift_id, t_near)
        else:
            self._add_lift_event(lift_id, label, t)

        #: Update datasources after modifying lift_event table
        self.update_datasource()

    def find_nearest_event(self, t, t_lim=1.):
        conn = create_engine(self.connection_string)

        query = '''
        SELECT * FROM lift_event
        WHERE lift_id = {id}
            AND timepoint BETWEEN {t1} AND {t2}
        '''.format(id=int(self.lift_select.value), t1=t-t_lim, t2=t+t_lim)

        #: Retrieve all lift_event items within the timeframe, if any
        events = read_sql(query, conn)

        if events.shape[0] > 0:
            # sort so that min(timepoint) comes first
            events = events.sort_values(by='timepoint', ascending=True).reset_index(drop=True)
            nearest_event = events.ix[0]  # returns a pandas Series
            # print 'Returning nearest event: \n{}'.format(nearest_event)
        else:
            nearest_event = None
            text = 'No events within {lim} seconds of tap at {t}'.format(lim=t_lim, t=t)
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

    def make_raw_plot(self, source, rep_start_source, rep_stop_source):
        tooltips = '''<div><span style="font-size: 12px;"> <b>time:</b> @timepoint s</span></div>
                      <div><span style="font-size: 12px;"> <b>accel:</b> @a_x_raw m/s^2</span></div>
                      <div><span style="font-size: 12px;"> <b>vel:</b> @v_x_raw m/s</span></div>
                      <div><span style="font-size: 12px;"> <b>pwr:</b> @p_x_raw W</span></div>'''

        #: Gold line that moves to where user clicks to show which timepoint will be logged on use of TapTool
        src = ColumnDataSource(
                dict(
                    x=[0., 0.],
                    y=[min(source.data['a_x']), max(source.data['a_x'])]
                )
            )

        draw_line_cb = CustomJS(
            args=dict(src=src, source=source),
            code="""

                // var path = document.location.pathname;
                // console.log(path);


                // package pg located at /usr/local/lib/node_modules/pg
                // var pg = require("/usr/local/lib/node_modules/pg");

                // var pg = require("pg");
                // console.log(require.paths);

                //var connectionString = "postgres://localhost:5432/fitai";
                //var pgClient = new pg.Client(connectionString);
                //pgClient.connect();
                //var query = pgClient.query("SELECT * FROM athlete_lift WHERE lift_id = 1");

                /*
                query.on("row", function(row, result){
                            result.addRow(row);
                        });

                pgClient.end()
                */

                // get data source from Callback args
                // var data = src.data;

                // bokeh TapTool callback object (cb_obj)
                // console.log(cb_obj);

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
            # y_range=Range1d(min(self.plot_source.data['a_rms']), max(self.plot_source.data['a_rms'])),
            y_range=Range1d(min(source.data['a_x']), max(source.data['a_x'])),
            plot_width=self.plot_width,
            plot_height=self.plot_height,
            h_symmetry=False,
            v_symmetry=False,
            min_border=0,
            toolbar_location='right',
            logo=None
        )

        rends = list()
        for y_val in ['a_x', 'v_x', 'p_x', 'a_x_hp', 'v_x_hp', 'p_x_hp']:
            #: Split out signal type (a/v/p) by color
            if 'a' in y_val:
                c = 'black'
            elif 'v' in y_val:
                c = 'blue'
            elif 'p' in y_val:
                c = 'purple'
            else:
                #: Uncaught line type here - make red so we can see it easily
                c = 'red'

            #: Differentiate between high-passed signal and non-HP signal
            if 'hp' in y_val:
                style = 'dashed'
            else:
                style = 'solid'

            l = Line(x='x_axis', y=y_val, name=y_val, line_color=c, line_dash=style, line_alpha=1)
            rend = GlyphRenderer(data_source=source, glyph=l, name=y_val)
            rends.append(rend)

            if y_val == 'a_x':
                #: Explicitly set the a_x glyph to be the basis of the hover tool
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

        # tap_lines = MultiLine(xs='xs', ys='ys', line_color=['red'], line_dash='dashed', line_width=2)
        # tap_lines_renderer = GlyphRenderer(data_source=src, glyph=tap_lines, name='tap_lines')

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

    def get_data(self, set_name):

        conn = create_engine(self.connection_string)

        if set_name == 'lift_options':
            query = '''
            SELECT
                al.lift_id,
                al.lift_type,
                ai.athlete_last_name || ', ' || ai.athlete_first_name AS athlete_name,
                ai.athlete_id
            FROM athlete_lift AS al
            INNER JOIN athlete_info AS ai
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
                print 'Key ({}, {}) found in storage dict'.format(int(self.lift_select.value), 'data')
                header = storage[(int(self.lift_select.value), 'header')]
                data = storage[(int(self.lift_select.value), 'data')]
                return header.copy(), data.copy()
            else:
                print 'Key ({}, {}) NOT found in storage dict'.format(int(self.lift_select.value), 'data')
                header, data = pull_data_by_lift(int(self.lift_select.value))

                for hp, lab in [(True, '_hp'), (False, '')]:
                    a_rms, v_rms, p_rms = process_data(header, data, RMS=True, highpass=hp)
                    accel, vel, pwr = process_data(header, data, RMS=False, highpass=hp)

                    #: If first loop, instantiate empty dataframe dat with proper index
                    if hp:
                        dat = DataFrame(index=a_rms.index)

                    for col in accel.columns:
                        raw_col = str(col) + '_raw' + lab
                        accel[raw_col] = accel[col]
                        accel[col+lab] = self.max_min_scale(accel[col])
                        #: If highpass, a_x will be present (cause the column won't be overwritten; a new column is
                        #: created), but we don't want to keep it.
                        if hp:
                            accel = accel.drop(col, axis=1)

                    for col in vel.columns:
                        raw_col = str(col) + '_raw' + lab
                        vel[raw_col] = vel[col]
                        vel[col+lab] = self.max_min_scale(vel[col])
                        if hp:
                            vel = vel.drop(col, axis=1)

                    for col in pwr.columns:
                        raw_col = str(col) + '_raw' + lab
                        pwr[raw_col] = pwr[col]
                        pwr[col+lab] = self.max_min_scale(pwr[col])
                        if hp:
                            pwr = pwr.drop(col, axis=1)

                    #: On first loop, dat should be empty dataframe with overlapping indices,
                    #: so these joins should be fine. On second loop, dat will already have half the data.
                    dat = dat.join(DataFrame(
                        data={
                            'a_rms'+lab: self.max_min_scale(a_rms),
                            'a_rms_raw'+lab: a_rms,
                            'v_rms'+lab: self.max_min_scale(v_rms),
                            'v_rms_raw'+lab: v_rms,
                            'p_rms'+lab: self.max_min_scale(p_rms),
                            'p_rms_raw'+lab: p_rms
                            },
                        index=a_rms.index).join(accel).join(vel).join(pwr))

                    #: Only want timepoint series once, so save until the end
                    if not hp:
                        dat = dat.join(data['timepoint'])

                # print dat.head()

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
    def format_lift_event_data(df, y_min, y_max, t_max):
        #: df is dataframe with columns lift_id, timepoint, event
        #: timepoint = time at which event occurs
        #: event = "rep_start" or "rep_stop"

        #: Bokeh MultiLine wants a series of points of the format
        #: [[x11, x12], [y11, y12]], [[x21, x22], [y21, y22]], ...
        df['x_val'] = df['timepoint']/float(t_max)  # scales all timepoints to be between 0 and 1 to fit on axis
        df['xs'] = df['x_val'].apply(lambda x: [x]*2)
        ys = [y_min, y_max]
        df['ys'] = [ys]*df.shape[0]  # replicate list "ys" df.shape[0] times

        df.drop(['x_val', 'timepoint', 'lift_id'], axis=1, inplace=True)

        #: Divvy up data according to whether each row reflects a rep start or stop
        starts = df.loc[df['event'] == 'rep_start'].drop('event', axis=1).reset_index(drop=True)
        stops = df.loc[df['event'] == 'rep_stop'].drop('event', axis=1).reset_index(drop=True)

        return starts, stops

    @staticmethod
    def max_min_scale(x):
        # return (x - min(x))/(max(x) - min(x))
        return x/(max(x) - min(x))

app = LiftPlot(conn_string, verbose=True)
curdoc().add_root(app.layout)
