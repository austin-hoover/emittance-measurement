from java.awt import BorderLayout
from java.awt import FlowLayout
from java.awt import Color
from java.awt.event import ActionEvent
from java.awt.event import ActionListener
from javax.swing import GroupLayout
from javax.swing import JComboBox
from javax.swing import JButton
from javax.swing import JLabel
from javax.swing import JPanel
from javax.swing import JTabbedPane
from javax.swing import JTextField


class PhaseController:
    
    def __init__(self, emitt_meas_document):
        self.emitt_meas_document = emitt_meas_document
        self.main_panel = JPanel(BorderLayout())
        
        
        test_button = JButton('Test')
        test_button.addActionListener(TestButtonListener(self))
        self.main_panel.add(test_button, BorderLayout.EAST)
        
#         energy_label = JLabel('Energy [GeV]')        
#         self.energy_text_field = JTextField()
#         self.energy_text_field.setForeground(Color.black)
#         self.left_panel.add(self.energy_text_field)
    
#         label1 = JLabel('mylabel1')
#         label2 = JLabel('mylabel2')
#         textfield1 = JTextField()
#         textfield2 = JTextField()
                
        panel = JPanel()  
        layout = GroupLayout(panel)
        panel.setLayout(layout);
        layout.setAutoCreateContainerGaps(True)
        layout.setAutoCreateGaps(True)
        
        group_labels = layout.createParallelGroup()
        group_fields = layout.createParallelGroup()
        group_rows = layout.createSequentialGroup()
        
        layout.setHorizontalGroup(layout.createSequentialGroup()
            .addGroup(group_labels)
            .addGroup(group_fields))
        layout.setVerticalGroup(group_rows)
                        
        def add_field(label, field):
            group_labels.addComponent(label)
            group_fields.addComponent(field)
            group_rows.addGroup(layout.createParallelGroup()
                .addComponent(label)
                .addComponent(field, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
        
        text_field_width = 12
        
        energy_label = JLabel('Energy [GeV]')
        energy_text_field = JTextField('1.0', text_field_width)
        add_field(energy_label, energy_text_field)
        
        ref_ws_id_label = JLabel('Ref. wire-scanner')
        ws_ids = ['RTBT_Diag:WS20', 'RTBT_Diag:WS21', 'RTBT_Diag:WS23', 'RTBT_Diag:WS24']
        ref_ws_id_dropdown = JComboBox(ws_ids);
        ref_ws_id_dropdown.setSelectedIndex(3);
        add_field(ref_ws_id_label, ref_ws_id_dropdown)
        
        phase_coverage_label = JLabel('Phase coverage [deg]')
        phase_coverage_text_field = JTextField('180.0', text_field_width)
        add_field(phase_coverage_label, phase_coverage_text_field)
        
        n_steps_label = JLabel('Total steps')
        n_steps_text_field = JTextField('12', text_field_width)
        add_field(n_steps_label, n_steps_text_field)
        
        max_beta_label = JLabel("<html>Max &beta; [m/rad]<html>")
        max_beta_text_field = JTextField('40.0', text_field_width)
        add_field(max_beta_label, max_beta_text_field)
            
        self.main_panel.add(panel, BorderLayout.WEST)


        
        
        
class TestButtonListener(ActionListener):
    def __init__(self, phase_controller):
        self.phase_controller = phase_controller

    def actionPerformed(self, actionEvent):
        return
#         text = self.phase_controller.energy_text_field.getText()
#         print text