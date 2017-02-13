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
from bokeh.models.glyphs import Line
from bokeh.models.tools import HoverTool, ResetTool, BoxZoomTool, PanTool, TapTool
from bokeh.layouts import Column, Row
from bokeh.models.widgets import CheckboxGroup, Button
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
        # all_dims = ['x', 'y', 'z', 'rms']
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

        self.signal_select = CheckboxGroup(name='signal_select',
                                           width=100,
                                           height=140,
                                           labels=['a', 'v', 'p', 'a_hp', 'v_hp', 'p_hp'],
                                           active=[0])
        self.signal_select.on_change('active', self._on_signal_change)

        #: Careful with this - button to delete all data associated with current lift
        #: For use in post-hoc data cleaning.
        self.del_button = Button(label='DELETE LIFT', button_type='danger', width=100, height=30)
        self.del_button.on_click(self._del_click)
        self.del_button_text = 'N/A'

    def _establish_outputs(self):
        # Has to be initialized before I can set the text.
        self.lift_info = Div(width=500, height=100)
        # To print success/fail
        self.del_button_info = Div(width=200, height=20)

    def _create_layout(self):

        self.del_header = Column(width=150, height=50)
        self.del_header.children = [self.del_button, self.del_button_info]

        self.plot_header = Row(width=self.plot_width, height=140)
        self.plot_header.children = [self.lift_select, self.signal_select, self.lift_info, self.del_header]

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
        self.raw_plot = self.make_raw_plot(self.plot_source)

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

        # The plot is set up to accept all values
        self.plot_source.data = ColumnDataSource(data).data
        self.plot_source.column_names = self.plot_source.data.keys()

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

    def make_raw_plot(self, source):
        tooltips = '''<div><span style="font-size: 12px;"> <b>time:</b> @timepoint s</span></div>
                      <div><span style="font-size: 12px;"> <b>accel:</b> @a_x_raw m/s^2</span></div>
                      <div><span style="font-size: 12px;"> <b>vel:</b> @v_x_raw m/s</span></div>
                      <div><span style="font-size: 12px;"> <b>pwr:</b> @p_x_raw W</span></div>'''

        #: Empty column data source for lines to be drawn onto plot
        src = ColumnDataSource(
            dict(
                # x=[max(source.data['x_axis'])/2., max(source.data['x_axis'])/2.],
                x=[0., 0.],
                y=[min(source.data['a_x']), max(source.data['a_x'])]
            )
        )

        draw_line_cb = CustomJS(
            args=dict(src=src, source=source),
            code="""
                // get data source from Callback args
                // var data = src.data;

                // bokeh TapTool callback object (cb_obj)
                // console.log(cb_obj);

                var data = cb_obj.data;
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

        #: Build renderer for lines generated by tap tool
        tap_line = Line(x='x', y='y', name='rep', line_color='green', line_dash='dashed', line_alpha=1., line_width=2)
        tap_line_renderer = GlyphRenderer(data_source=src, glyph=tap_line, name='tap_line')

        rends.append(tap_line_renderer)

        tap_rends = [tap_line_renderer] + h_rends
        taptool = TapTool(renderers=tap_rends, callback=draw_line_cb)

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

    @staticmethod
    def max_min_scale(x):
        # return (x - min(x))/(max(x) - min(x))
        return x/(max(x) - min(x))

app = LiftPlot(conn_string, verbose=True)
curdoc().add_root(app.layout)