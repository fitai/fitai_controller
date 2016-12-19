from sqlalchemy import create_engine
from pandas import read_sql, DataFrame
from numpy import where as np_where
from os.path import dirname, abspath
import sys

try:
    sys.path.append(dirname(dirname(abspath(__file__))))
except NameError:
    print 'Working in dev mode.'

from databasing.db_conn_strings import aws_conn_string
from databasing.database_pull import pull_data_by_lift

from bokeh.models import Plot, ColumnDataSource, GlyphRenderer
from bokeh.models.ranges import Range1d
from bokeh.models.glyphs import Line
from bokeh.models.tools import HoverTool, ResetTool, BoxZoomTool, PanTool
from bokeh.layouts import Column, Row
from bokeh.models.widgets import CheckboxGroup
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.plotting import curdoc

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
        self.main_plot = Plot()

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
                                         labels=['a_rms', 'v_rms', 'p_rms'],
                                         active=[1, 1, 1])
        self.signal_select.on_change('active', self._on_lift_change)

        # Has to be initialized before I can set the text.
        # TODO: Consider rearranging order of element creation and layout of app
        self.info = Div(width=600, height=100)

        # Specifies what goes where in the app
        self._load_content()
        self._create_layout()

    def _create_layout(self):

        self.plot_header = Row(width=950, height=100)
        self.plot_header.children = [self.lift_select, self.signal_select, self.info]

        self.layout = Row(children=[self.plot_header, self.main_plot], width=950, height=500)

    def _load_content(self):
        self.update_datasource()
        self.main_plot = self.make_plot(self.plot_source, self.plot_alphas.data)

    def _on_lift_change(self, attr, old, new):
        # if attr['name'] == 'lift_id':
        #     print 'Updating plot with lift_id {}'.format(new)
        self.update_datasource()

    def update_datasource(self):

        header, data = self.get_data('lift_data')

        self.info.text = str(header.ix[0].to_json())

        # Leaving separated for readability
        y = np_where([x==1 for x in self.signal_select.active])[0]
        cols = [self.signal_select.labels[x] for x in y]  # Extract all labels that are active

        print 'active columns: {}'.format(cols)
        # By default, all signals must be plotted because I only want to update the ColumnDataSource, not
        # rebuild the plot N times. To accommodate this, I will just update the alpha of the lines
        alphas = DataFrame(data={'a_rms': 0., 'v_rms': 0., 'p_rms': 0.}, index=[0])
        for col in cols:
            alphas[col] = 1.

        self.plot_alphas = ColumnDataSource(alphas)

        # The plot is set up to accept all values
        self.plot_source = ColumnDataSource(data)

        print 'done updating plot datasource'
        # self.main_plot = self.make_plot(self.plot_source, self.plot_alphas.data)

    def make_plot(self, source, alphas):
        tooltips = '''<div><span style="font-size: 12px;"> <b>time:</b> @timepoint s</span></div>
                      <div><span style="font-size: 12px;"> <b>accel:</b> @a_rms m/s^2</span></div>
                      <div><span style="font-size: 12px;"> <b>vel:</b> @v_rms m/s</span></div>
                      <div><span style="font-size: 12px;"> <b>pwr:</b> @p_rms W</span></div>'''

        plot = Plot(
            title=None,
            x_range=Range1d(min(self.plot_source.data['timepoint']), max(self.plot_source.data['timepoint'])),
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

        # axis_theme = dict(
        #     axis_label=None, axis_label_text_font_size='0pt', minor_tick_line_alpha=0.0,
        #     axis_line_alpha=0.0, major_tick_line_alpha=0.0, major_label_text_color='grey',
        #     major_label_text_font_size='0pt')
        #
        # plot.add_layout(LinearAxis(**axis_theme), 'left')
        # plot.add_layout(LinearAxis(**axis_theme), 'below')

        # Note: ColumnDataSource.data[key[0]] returns a list. Want the value in that list (there should be only 1)
        a_line = Line(x='timepoint', y='a_rms', name='accel', line_color='black', line_alpha=alphas['a_rms'][0])
        a_line_renderer = GlyphRenderer(data_source=source, glyph=a_line, name='accel_rend')

        v_line = Line(x='timepoint', y='v_rms', name='vel', line_color='blue', line_alpha=alphas['v_rms'][0])
        v_line_renderer = GlyphRenderer(data_source=source, glyph=v_line, name='vel_rend')

        p_line = Line(x='timepoint', y='p_rms', name='pwr', line_color='purple', line_alpha=alphas['p_rms'][0])
        p_line_renderer = GlyphRenderer(data_source=source, glyph=p_line, name='pwr_rend')

        hover = HoverTool(renderers=[a_line_renderer, v_line_renderer, p_line_renderer], tooltips=tooltips, point_policy='follow_mouse')
        zoom = BoxZoomTool()
        reset = ResetTool()
        pan = PanTool()
        plot.renderers.extend([a_line_renderer, v_line_renderer, p_line_renderer])
        plot.tools.extend([hover, zoom, reset, pan])
        # TODO figure out legend
        plot.legend.append([a_line])
        plot.legend.location = 'upper_left'

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
        elif set_name == 'users':
            print 'Not implemented.'
        elif set_name == 'lift_data':
            if (int(self.lift_select.value), 'data') in storage.keys():
                header = storage[(int(self.lift_select.value), 'header')]
                data = storage[(int(self.lift_select.value), 'data')]
                return header, data
            else:
                header, data = pull_data_by_lift(int(self.lift_select.value))
                a, v, p = process_data(header, data)
                dat = DataFrame(data={'a_rms': self.max_min_scale(a),
                                      'v_rms': self.max_min_scale(v),
                                      'p_rms': self.max_min_scale(p),
                                      'timepoint': data['timepoint']},
                                index=a.index)
                storage[(int(self.lift_select.value), 'data')] = dat
                storage[(int(self.lift_select.value), 'header')] = header
                return header, dat

        values = read_sql(query, conn)

        return values

    @staticmethod
    def max_min_scale(x):
        return (x - min(x))/(max(x) - min(x))

app = LiftPlot(aws_conn_string, verbose=True)
curdoc().add_root(app.layout)
