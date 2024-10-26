import os
from EventToSalienceMap import EventToSalienceMap

event = EventToSalienceMap()
data_dir = r"C:\Users\maxeb\Documents\Radboud\Master\Master Thesis\Dataset tests\MasterThesisDataSet"
ignore_file = os.path.join(data_dir, "ignore.txt")
event.train_on_centre_SNN(data_dir, ignore_file)