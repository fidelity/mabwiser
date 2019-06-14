#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest


# Test Directory
start_dir = '.'

# Test Loader
loader = unittest.TestLoader()

# Test Suite
suite = loader.discover(start_dir)

# Test Runner
runner = unittest.TextTestRunner()

# Run the Suite
runner.run(suite)
