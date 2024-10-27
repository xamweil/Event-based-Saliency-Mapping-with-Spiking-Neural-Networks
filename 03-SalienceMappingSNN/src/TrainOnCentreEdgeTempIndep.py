import os
from EventToSalienceMap import EventToSalienceMap

event = EventToSalienceMap()
data_dir = r"D:\MasterThesisDataSet"
ignore_file = os.path.join(data_dir, "ignore.txt")
(event.train_on_centre_SNN_tempo_independent(data_dir, ignore_file))