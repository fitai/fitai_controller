from sqlalchemy import create_engine
from pandas import read_sql, DataFrame
from numpy import where as np_where, tile, repeat
from os.path import dirname, abspath
from sys import path as sys_path

try:
    sys_path.append(dirname(dirname(abspath(__file__))))
except NameError:
    print 'Working in dev mode.'

from bokeh.models import Plot, ColumnDataSource, GlyphRenderer
from bokeh.models.widgets.panels import Tabs, Panel
from bokeh.models.ranges import Range1d
from bokeh.models.glyphs import Line
from bokeh.models.tools import HoverTool, ResetTool, BoxZoomTool, PanTool
from bokeh.layouts import Column, Row
from bokeh.models.widgets import CheckboxGroup
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.plotting import curdoc

from databasing.db_conn_strings import local_conn_string as aws_conn_string
from databasing.database_pull import pull_data_by_lift
from processing.util import process_data


storage = dict()


class LiftPlot(object):

    def __init__(self, connection_string, verbose=True):

        self.connection_string = connection_string
        self.verbose = verbose

        # Elements of the bokeh app that will have to be laid out later
        lift_options = [str(x) for x in sorted(self.get_data('lift_options')['lift_id'].unique())]

        self.plot_source = ColumnDataSource()
        self.plot_alphas = ColumnDataSource()

        self.rms_plot = Plot()
        self.raw_plot = Plot()

        self.raw_dims = list()

        # ### Effectively a global value - all possible signals that could be passed in ###
        #: By default, all signals must be plotted because I only want to update the ColumnDataSource, not
        #: rebuild the plot N times. To accommodate this, I will just update the alpha of the lines.
        #: Start by defaulting all lines to alpha = 0
        # all_dims = ['x', 'y', 'z', 'rms']
        all_dims = ['x', 'rms']
        all_cols = ['a', 'v', 'p']
        all_opts = [x+'_'+y for (x, y) in zip(tile(all_cols, len(all_dims)), tile(all_dims, len(all_cols)))]
        self.all_signals = all_opts
        self.active_signals = list()  # to be filled in each time update_datasource() is called

        # ###

        #: TODO Move these inputs out of the __init__ method
        self.lift_select = Select(name='lift_id',
                                  title='Select lift_id',
                                  width=100,
                                  options=lift_options,
                                  value=lift_options[0])
        # Explicitly set callback after creation - can't do it in creation step??
        self.lift_select.on_change('value', self._on_lift_change)

        self.signal_select = CheckboxGroup(name='signal_select',
                                         width=70,
                                         labels=['a', 'v', 'p'],
                                         active=[1, 1, 0])
        self.signal_select.on_change('active', self._on_lift_change)

        # Has to be initialized before I can set the text.
        # TODO: Consider rearranging order of element creation and layout of app
        self.info = Div(width=500, height=100)

        # Specifies what goes where in the app
        self._load_content()
        self._create_layout()

    def _create_layout(self):

        self.plot_header = Row(width=950, height=100)
        self.plot_header.children = [self.lift_select, self.signal_select, self.info]

        ## RMS PLOT ##

        # Box that contains the RMS plot (Box may be unnecessary)
        self.rms_panel_box = Column(width=900, height=600)
        self.rms_panel_box.children = [self.rms_plot]

        # Panel that contains the RMS box
        self.panel_rms = Panel(
            child=self.rms_plot, title='RMS Plot', closable=False, width=900, height=600)

        ## RAW PLOT ##
        self.raw_panel_box = Column(width=900, height=600)
        self.raw_panel_box.children = [self.raw_plot]

        self.panel_raw = Panel(
            child=self.raw_plot, title='Raw Plot', closable=False, width=900, height=600)

        # Contains ALL panels
        self.panel_parent = Tabs(width=950, height=600, active=1)

        self.panel_parent.tabs = [self.panel_rms, self.panel_raw]

        self.layout = Row(children=[self.plot_header, self.panel_parent], width=950, height=500)

    def _load_content(self):
        self.update_datasource()
        self.rms_plot = self.make_RMS_plot(self.plot_source)
        self.raw_plot = self.make_raw_plot(self.plot_source)

    def _on_lift_change(self, attr, old, new):
        print attr
        # if attr['name'] == 'lift_id':
        #     print 'Updating plot with lift_id {}'.format(new)
        self.update_datasource()

    def update_datasource(self):

        header, data = self.get_data('lift_data')

        # Lift metadata to display
        self.info.text = str(header.ix[0].to_json())

        # It's unknown ahead of time which raw dimensions will be present in each lift, so we need to find
        # them all dynamically. We known RMS will be present for a, v, p.
        raw_headers = [x.split('_')[-1] for x in data.columns if ('rms' not in x) and ('time' not in x)]
        self.raw_dims = list(set(raw_headers))  # remove duplicates

        # Leaving separated for readability
        y = np_where([x == 1 for x in self.signal_select.active])[0]
        cols = [self.signal_select.labels[x].split('_')[0] for x in y]  # Extract all labels that are active

        print 'active columns: {}'.format(cols)

        #: Change whichever columns have been selected (along appropriate dimensions) to alpha = 1
        dims = self.raw_dims + ['rms']
        selected = [x+'_'+y for (x, y) in zip(repeat(cols, len(dims)), tile(dims, len(cols)))]

        self.active_signals = selected

        alphas_dict = {k: 0 for k in self.all_signals}
        alphas_update = {k: 1 for k in selected}
        alphas_dict.update(alphas_update)

        for col in alphas_dict:
            col_name = str(col) + '_alpha'
            data[col_name] = [alphas_dict[col]] * data.shape[0]

        #: TODO work in the alphas such that the lines actually disappear

        # To draw on y=0; just for now (probably)
        data['zero'] = 0.
        data['x_axis'] = data['timepoint']/max(data['timepoint'])  # scale x axis to be (0, 1)

        # The plot is set up to accept all values
        self.plot_source.data = ColumnDataSource(data).data
        self.plot_source.column_names = self.plot_source.data.keys()

        # self.plot_alphas.data = ColumnDataSource(DataFrame(data=alphas_dict, index=[0])).data
        # self.plot_alphas.column_names = self.plot_alphas.data.keys()

        self.rms_plot = self.make_raw_plot(self.plot_source)
        print 'done updating plot datasource'

    def make_RMS_plot(self, source):
        tooltips = '''<div><span style="font-size: 12px;"> <b>time:</b> @timepoint s</span></div>
                      <div><span style="font-size: 12px;"> <b>accel:</b> @a_rms m/s^2</span></div>
                      <div><span style="font-size: 12px;"> <b>vel:</b> @v_rms m/s</span></div>
                      <div><span style="font-size: 12px;"> <b>pwr:</b> @p_rms W</span></div>'''

        plot = Plot(
            title=None,
            x_range=Range1d(min(source.data['x_axis']), max(source.data['x_axis'])),
            # y_range=Range1d(min(self.plot_source.data['a_rms']), max(self.plot_source.data['a_rms'])),
            y_range=Range1d(0, 1),
            plot_width=1000,
            plot_height=600,
            h_symmetry=False,
            v_symmetry=False,
            min_border=0,
            toolbar_location='right',
            logo=None
        )

        rends = list()
        for y_val in [x for x in self.all_signals if 'rms' in x]:
            y_alpha = y_val + '_alpha'

            if 'a' in y_val:
                c = 'black'
            elif 'v' in y_val:
                c = 'blue'
            elif 'p' in y_val:
                c = 'purple'
            else:
                c = 'red'

            print 'building line glyph for signal {s} (alpha={a})'.format(s=y_val, a=max(source.data[y_alpha]))

            l = Line(x='x_axis', y=y_val, name=y_val, line_color=c, line_alpha=max(source.data[y_alpha]))
            rends.append(GlyphRenderer(data_source=source, glyph=l, name=y_val))

        # axis_theme = dict(
        #     axis_label=None, axis_label_text_font_size='0pt', minor_tick_line_alpha=0.0,
        #     axis_line_alpha=0.0, major_tick_line_alpha=0.0, major_label_text_color='grey',
        #     major_label_text_font_size='0pt')
        #
        # plot.add_layout(LinearAxis(**axis_theme), 'left')
        # plot.add_layout(LinearAxis(**axis_theme), 'below')

        # Note: ColumnDataSource.data[key[0]] returns a list. Want the value in that list (there should be only 1)
        # a_line = Line(x='x_axis', y='a_rms', name='accel', line_color='black', line_alpha=1)
        # a_line_renderer = GlyphRenderer(data_source=source, glyph=a_line, name='accel_rend')
        #
        # v_line = Line(x='x_axis', y='v_rms', name='vel', line_color='blue', line_alpha=1)
        # v_line_renderer = GlyphRenderer(data_source=source, glyph=v_line, name='vel_rend')
        #
        # p_line = Line(x='x_axis', y='p_rms', name='pwr', line_color='purple', line_alpha=0)
        # p_line_renderer = GlyphRenderer(data_source=source, glyph=p_line, name='pwr_rend')

        hover = HoverTool(renderers=rends, tooltips=tooltips, point_policy='follow_mouse')
        zoom = BoxZoomTool()
        reset = ResetTool()
        pan = PanTool()
        plot.renderers.extend(rends)
        plot.tools.extend([hover, zoom, reset, pan])
        # TODO figure out legend
        # plot.legend.append([a_line])
        # plot.legend.location = 'upper_left'

        return plot

    def make_raw_plot(self, source):
        tooltips = '''<div><span style="font-size: 12px;"> <b>time:</b> @timepoint s</span></div>
                      <div><span style="font-size: 12px;"> <b>accel:</b> @a_x m/s^2</span></div>
                      <div><span style="font-size: 12px;"> <b>vel:</b> @v_x m/s</span></div>
                      <div><span style="font-size: 12px;"> <b>pwr:</b> @p_x W</span></div>'''

        plot = Plot(
            title=None,
            x_range=Range1d(min(source.data['x_axis']), max(source.data['x_axis'])),
            # y_range=Range1d(min(self.plot_source.data['a_rms']), max(self.plot_source.data['a_rms'])),
            y_range=Range1d(min(source.data['a_x']), max(source.data['a_x'])),
            plot_width=1000,
            plot_height=600,
            h_symmetry=False,
            v_symmetry=False,
            min_border=0,
            toolbar_location='right',
            logo=None
        )

        # axis_theme = dict(
        #     axis_label=None, axis_label_text_font_size='0pt', minor_tick_line_alpha=0.0,
        #     axis_line_alpha=0.0, major_tick_line_alpha=0.0, major_label_text_color='grey',
        #     major_label_text_font_size='0pt')
        #
        # plot.add_layout(LinearAxis(**axis_theme), 'left')
        # plot.add_layout(LinearAxis(**axis_theme), 'below')

        # Note: ColumnDataSource.data[key[0]] returns a list. Want the value in that list (there should be only 1)

        rends = list()

        for y_val in ['a_x', 'v_x', 'p_x']:
            y_alpha = y_val + '_alpha'

            if 'a' in y_val:
                c = 'black'
            elif 'v' in y_val:
                c = 'blue'
            elif 'p' in y_val:
                c = 'purple'
            else:
                c = 'red'

            print 'building line glyph for signal {s} (alpha={a})'.format(s=y_val, a=max(source.data[y_alpha]))

            l = Line(x='x_axis', y=y_val, name=y_val, line_color=c, line_alpha=max(source.data[y_alpha]))
            rends.append(GlyphRenderer(data_source=source, glyph=l, name=y_val))
            # plot.renderers.extend([GlyphRenderer(data_source=source, glyph=l, name=y_val)])

        # a_line = Line(x='timepoint', y='a_x', name='accel', line_color='black', line_alpha=1)
        # a_line_renderer = GlyphRenderer(data_source=source, glyph=a_line, name='accel_rend')
        #
        # v_line = Line(x='timepoint', y='v_x', name='vel', line_color='blue', line_alpha=1)
        # v_line_renderer = GlyphRenderer(data_source=source, glyph=v_line, name='vel_rend')
        #
        # p_line = Line(x='timepoint', y='p_x', name='pwr', line_color='purple', line_alpha=0)
        # p_line_renderer = GlyphRenderer(data_source=source, glyph=p_line, name='pwr_rend')
        #
        zero_line = Line(x='x_axis', y='zero', name='zero', line_color='red', line_dash='dashed', line_alpha=1)
        zero_line_renderer = GlyphRenderer(data_source=source, glyph=zero_line, name='zero_rend')

        rends.append(zero_line_renderer)
        # plot.renderers.extend([zero_line_renderer])

        hover = HoverTool(
            renderers=rends,
            tooltips=tooltips, point_policy='follow_mouse')
        zoom = BoxZoomTool()
        reset = ResetTool()
        pan = PanTool()
        plot.renderers.extend(rends)
        plot.tools.extend([hover, zoom, reset, pan])
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
                a_rms, v_rms, p_rms = process_data(header, data, RMS=True)
                accel, vel, pwr = process_data(header, data, RMS=False)
                dat = DataFrame(data={'a_rms': self.max_min_scale(a_rms),
                                      'v_rms': self.max_min_scale(v_rms),
                                      'p_rms': self.max_min_scale(p_rms),
                                      'timepoint': data['timepoint']},
                                index=a_rms.index).join(accel).join(vel).join(pwr)
                storage[(int(self.lift_select.value), 'data')] = dat
                storage[(int(self.lift_select.value), 'header')] = header
                return header.copy(), dat.copy()

    @staticmethod
    def max_min_scale(x):
        return (x - min(x))/(max(x) - min(x))

app = LiftPlot(aws_conn_string, verbose=True)
curdoc().add_root(app.layout)
