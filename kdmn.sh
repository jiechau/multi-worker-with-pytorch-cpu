#!/bin/bash
ps -ef | grep dmn | grep -v grep | grep -v kdmn | awk '{print $2}' | xargs kill -9
