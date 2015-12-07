# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:48:15 2015

@author: wt
"""

import csv

userIdMap = {}

with open('mrredges-no-tweet-no-retweet-poi-counted.csv', 'rt') as f:
    reader = csv.reader(f)
    first_row = next(reader)
    for row in reader:
        u1 = row[0]
        u2 = row[1]
        btype = row[2]
        count = row[3]
        userID = userIdMap.get(u1,len(userIdMap))
        userIdMap[u1] = userID
        userID = userIdMap.get(u2,len(userIdMap))
        userIdMap[u2] = userID
        print u1, u2, btype, count