# PRAP-PIM

Download cifar10 data.

Run train() function, and it will train an original model.  

Predefine prune ratio and reuse ratio for each model. 

Run get_structure_mask(), get_ORC_mask(), or get_shape_mask() function, it will return the pruning information.

Run pattern_value_identical_translate(), pattern_value_similar_translate(), structure_and_value_identical_translate(), or pattern_shape_and_value_similar_translate() function, and it will return the map information and actual reuse ratio.

Run pattern_translate function, and it will fine-tune the weights.

Use the prune ratio and actual reuse ratio fill the prune_config_list[] and reuse_config_list[], respectively, for each model in simulator/interface/network.py. 

Run Model_latency.py and Model_energy.py.
