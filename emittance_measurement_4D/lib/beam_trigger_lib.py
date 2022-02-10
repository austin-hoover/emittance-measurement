"""Beam trigger library.

This was ported from a different application.
"""
from __future__ import print_function
import time
from java.lang import *  # Wildcard imports are not allowed in Java 17+
from xal.extension.scan import WrappedChannel


class BeamTrigger:
    """Class to trigger the beam.

    Note: if the there is an error in the machine, this class will not trigger
    the beam. It seems that manually triggering the beam from StartMap will solve
    the problem.
    """

    def __init__(self):
        self.beam_trigger_wpv = None
        self.test_pv = None
        self.sleep_time = 0.1
        self.fake_scan = False
        self.use_trigger = True

    def init_channels(self):
        if not self.fake_scan:
            if self.beam_trigger_wpv is None or self.test_pv is None:
                self.beam_trigger_wpv = WrappedChannel("ICS_Tim:Gate_BeamOn:SSTrigger")
                self.test_pv = WrappedChannel("MEBT_Diag:BPM01:amplitudeAvg")
                self.test_pv.startMonitor()
                time.sleep(2.0)

    def set_sleep_time(self, sleep_time):
        self.sleep_time = sleep_time

    def get_sleep_time(self):
        return self.sleep_time

    def set_fake_scan(self, bool_val):
        self.fake_scan = bool_val

    def get_fake_scan(self):
        return self.fake_scan

    def set_use_trigger(self, bool_val):
        self.use_trigger = bool_val

    def get_use_trigger(self):
        return self.use_trigger

    def fire(self):
        print("Making shot...")
        self.init_channels()

        if not self.fake_scan:
            self.test_pv.setValueChanged(False)
            time.sleep(0.01)
            if self.use_trigger:
                print("triggering...")
                self.beam_trigger_wpv.setValue(1.0)
                time.sleep(0.5)
                self.beam_trigger_wpv.setValue(0.0)
                print("trigger value = {0}".format(self.beam_trigger_wpv.getValue()))

        time.sleep(self.sleep_time)

        count = 0
        time_sleep = 0.05
        if not self.fake_scan:
            while not self.test_pv.valueChanged():
                count += 1
                time.sleep(time_sleep)
                if count % 25 == 0:
                    time_sleep += 0.1
                if count > 25:
                    print(
                        "Something is wrong! Please fire the beam manually! Bad count = {}".format(
                            count
                        )
                    )
        return True
