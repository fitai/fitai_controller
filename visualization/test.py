from sqlalchemy import create_engine
from pandas import read_sql
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
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.plotting import curdoc


storage = dict()


class LiftPlot(object):

    def __init__(self, connection_string, verbose=True):

        self.connection_string = connection_string
        self.verbose = verbose

        # Elements of the bokeh app that will have to be laid out later
        lift_options = [str(x) for x in sorted(self.get_data('lift_options')['lift_id'].unique())]

        self.plot_source = ColumnDataSource()
        self.main_plot = Plot()
        self.lift_select = Select(name='lift_id',
                                  title='Select lift_id',
                                  width=100,
                                  options=lift_options,
                                  value=lift_options[0])
        # Explicitly set callback after creation - can't do it in creation step??
        self.lift_select.on_change('value', self._on_lift_change)

        # Has to be initialized before I can set the text.
        # TODO: Consider rearranging order of element creation and layout of app
        self.info = Div(width=700, height=100)

        # Specifies what goes where in the app
        self._load_content()
        self._create_layout()

    def _create_layout(self):

        self.plot_header = Column(width=950, height=100)
        # TODO: Figure out why lift_select and info Div won't appear side by side :-/
        self.plot_header.children = [self.lift_select, self.info]

        self.layout = Row(children=[self.plot_header, self.main_plot], width=950, height=500)

    def _load_content(self):
        self.update_datasource()
        self.main_plot = self.make_plot(self.plot_source)

    def _on_lift_change(self, attr, old, new):
        print 'Updating plot with lift_id {}'.format(new)
        self.update_datasource()

    def update_datasource(self):

        header, data = self.get_data('lift_data')

        dat = data.drop('lift_id', axis=1)

        self.info.text = str(header.ix[0].to_json())
        # print 'data contains: \n{}'.format(dat.head())

        self.plot_source.data = ColumnDataSource(dat).data
        self.plot_source.column_names = self.plot_source.data.keys()

        print 'done updating plot datasource'

    def make_plot(self, ds):
        tooltips = '''<div><span style="font-size: 12px;"> <b>time:</b> @timepoint s</span></div>
                      <div><span style="font-size: 12px;"> <b>accel:</b> @a_x m/s^2</span></div> '''

        plot = Plot(
            title=None,
            x_range=Range1d(min(self.plot_source.data['timepoint']), max(self.plot_source.data['timepoint'])),
            y_range=Range1d(min(self.plot_source.data['a_x']), max(self.plot_source.data['a_x'])),
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

        line = Line(x='timepoint', y='a_x', line_color='black')
        line_renderer = GlyphRenderer(data_source=ds, glyph=line)

        hover = HoverTool(renderers=[line_renderer], tooltips=tooltips, point_policy='follow_mouse')
        zoom = BoxZoomTool()
        reset = ResetTool()
        pan = PanTool()
        plot.renderers.extend([line_renderer])
        plot.tools.extend([hover, zoom, reset, pan])

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
            header, data = pull_data_by_lift(int(self.lift_select.value))
            storage[header['lift_id'].ix[0]] = data
            return header, data

        values = read_sql(query, conn)

        return values

app = LiftPlot(aws_conn_string, verbose=True)
curdoc().add_root(app.layout)
