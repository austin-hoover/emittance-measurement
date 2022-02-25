import sys

from java.awt import BorderLayout
from java.awt import Dimension
from java.awt.event import WindowAdapter
from javax.swing import JFrame
from javax.swing import JPanel
from javax.swing import JTabbedPane

from lib.analysis_panel import AnalysisPanel
from lib.phase_controller_panel import PhaseControllerPanel
from lib.time_and_date_lib import DateAndTimeText


class EmittanceMeasurement4D:
    def __init__(self):
        # Create panels.
        self.phase_controller_panel = PhaseControllerPanel()
        self.analysis_panel = AnalysisPanel()

        # Add panels to tabbed pane.
        self.pane = JTabbedPane(JTabbedPane.TOP)
        self.pane.addTab("Phase controller", self.phase_controller_panel)
        self.pane.addTab("Analysis", self.analysis_panel)

        # Add tabbed pane to frame.
        self.frame = JFrame("4D Emittance Measurement")
        self.frame.setSize(Dimension(1475, 1150))
        self.frame.add(self.pane)

        # Add time stamp at the bottom of the frame.
        time_text = DateAndTimeText()
        time_panel = JPanel(BorderLayout())
        time_panel.add(time_text.getTimeTextField(), BorderLayout.CENTER)
        self.frame.add(time_panel, BorderLayout.SOUTH)

    def launch(self):
        class WindowCloser(WindowAdapter):
            def __init__(self):
                return
            def windowClosing(self, event):
                sys.exit(1)

        self.frame.addWindowListener(WindowCloser())
        self.frame.show()


if __name__ == "__main__":
    EmittanceMeasurement4D().launch()
