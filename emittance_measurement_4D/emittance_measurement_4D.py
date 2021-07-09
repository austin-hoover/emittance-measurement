import sys
import math
import types
import time
import random
import os

from java.lang import *
from javax.swing import *
from java.awt import BorderLayout
from java.awt import Color
from java.awt import Dimension
from java.awt.event import WindowAdapter
from java.beans import PropertyChangeListener
from java.awt.event import ActionListener
from java.util import ArrayList
from java.io import File
from java.net import URL

from xal.extension.application import XalDocument
from xal.extension.application import ApplicationAdaptor
from xal.extension.application.smf import AcceleratorApplication
from xal.extension.application.smf import AcceleratorDocument
from xal.extension.application.smf import AcceleratorWindow
from xal.smf import AcceleratorSeqCombo
from xal.smf.data import XMLDataManager

from lib.time_and_date_lib import DateAndTimeText
from lib.phase_controller import PhaseController



# Local Classes that are not subclasses of XAL Accelerator Framework
#------------------------------------------------------------------------------

class EmittMeas_Window:

    def __init__(self, empty_document): 
        # `empty_document` is the parent document for all controllers
        self.empty_document = empty_document
        self.frame = None
        self.center_panel = JPanel(BorderLayout())
        self.main_panel = JPanel(BorderLayout())
        self.time_txt = DateAndTimeText()

        # Add time and text panels
        time_panel = JPanel(BorderLayout())
        time_panel.add(self.time_txt.getTimeTextField(), BorderLayout.CENTER)
        self.center_panel.add(self.main_panel, BorderLayout.CENTER)
#         tmp_panel = JPanel(BorderLayout())
#         tmp_panel.add(time_panel, BorderLayout.WEST)
        self.center_panel.add(time_panel, BorderLayout.SOUTH)

    def setFrame(self, xal_frame, main_panel):
        self.frame = xal_frame
        main_panel.add(self.center_panel, BorderLayout.CENTER)

    def getMainPanel(self):
        return self.main_panel

    def getMessageTextField(self):
        return self.message_text_field

    
class EmittMeas_Document:
    """Put all logic and GUI in this class."""
    def __init__(self):
        self.phase_controller = PhaseController(self)
        
        self.empty_window = None
        self.tabbed_pane = JTabbedPane()
        self.tabbed_pane.add('RTBT Phase Controller', self.phase_controller.main_panel)

    def setWindow(self,empty_window):
        self.empty_window = empty_window
        self.empty_window.getMainPanel().add(self.tabbed_pane, BorderLayout.CENTER)

    def getWindow(self):
        return self.empty_window

    def getMessageTextField(self):
        if(self.empty_window != None):
            return self.empty_window.getMessageTextField()
        else:
            return None


# SUBCLASSES of XAL Accelerator Framework
#------------------------------------------------------------------------------
class EmittMeas_OpenXAL_Document(AcceleratorDocument):
    def __init__(self,url = None):
        self.main_panel = JPanel(BorderLayout())

        #==== set up accelerator 
        if(not self.loadDefaultAccelerator()):
            self.applySelectedAcceleratorWithDefaultPath("/default/main.xal")

        self.empty_document = EmittMeas_Document()
        self.empty_window = EmittMeas_Window(self.empty_document)

        if(url != None):
            self.setSource(url)
            self.readEmittMeas_Document(url)
            #super class method - will show "Save" menu active
            if(url.getProtocol().find("jar") >= 0):
                self.setHasChanges(False)
            else:
                self.setHasChanges(True)

    def makeMainWindow(self):
        self.mainWindow = EmittMeas_OpenXAL_Window(self)
        self.mainWindow.getContentPane().setLayout(BorderLayout())
        self.mainWindow.getContentPane().add(self.main_panel, BorderLayout.CENTER)
        self.empty_window.setFrame(self.mainWindow, self.main_panel)
        self.empty_document.setWindow(self.empty_window)
        self.mainWindow.setSize(Dimension(800, 600))

    def saveDocumentAs(	self,url):
        # here you save of the application to the XML file 
        pass

    def readEmittMeas_Document(self,url):
        # here you put the initialization of the application from the XML file 
        pass


class EmittMeas_OpenXAL_Window(AcceleratorWindow):
    def __init__(self,empty_openxal_document):
        AcceleratorWindow.__init__(self,empty_openxal_document)

    def getMainPanel(self):
        return self.document.main_panel

    
class EmittMeas_OpenXAL_Main(ApplicationAdaptor):
    def __init__(self):
        ApplicationAdaptor.__init__(self)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.setResourcesParentDirectoryWithPath(script_dir)

    def readableDocumentTypes(self):
        return ["empty",]

    def writableDocumentTypes(self):
        return self.readableDocumentTypes()

    def newEmptyDocument(self, *args):
        if len( args ) > 0:
            return ApplicationAdaptor.newEmptyDocument(self,*args)
        else:
            return self.newDocument(None)

    def newDocument(self,location):
        return EmittMeas_OpenXAL_Document(location)

    def applicationName(self):
        return "4D Emittance Measurement"

AcceleratorApplication.launch(EmittMeas_OpenXAL_Main())