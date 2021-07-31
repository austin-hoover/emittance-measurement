rec_point_label = JLabel('Reconstruction point')

nodes = phase_controller.sequence.getNodes()
node_ids = [node.getId() for node in nodes]
rec_point_dropdown = JComboBox(node_ids)