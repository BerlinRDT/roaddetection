#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for network models
"""
import unittest
import numpy as np
from src.models.network_models import unet_flex


class dimred_test(unittest.TestCase):
    def setUp(self):
        # 3-band image 512 by 512, channels-first
        self.test_img_3band = np.random.randn(3, 512, 512)

    def test_architecture(self):
        # first hurdle: models must compile
        # - defaults
        model_list = [unet_flex()]
        # depth
        for depth in range(1, 7):
            model_list.append(unet_flex(num_level=depth))
        # number of initial filters at default depth
        for fb in range(1, 10):
            model_list.append(unet_flex(num_filt_init=2**fb))
        
        print("\n\ntest-compiling {} models...".format(len(model_list)))
        for m in model_list:
            m.compile(optimizer="RMSprop", loss="binary_crossentropy", metrics=["accuracy"])

    def test_output_dimensions(self):
        # ensure that input and output size are identical
        # ... to be continued
        return None

        
if __name__ == '__main__':
    unittest.main()