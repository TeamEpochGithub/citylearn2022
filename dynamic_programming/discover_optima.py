# Select an hour as entry point
# Split the current action space and recursively explore them
# At each recursive step, the hour is incremented and the action space of that hour is split.
# When the required depth is reached, the current hour is evaluated and the results are pushed upwards.
# At each split, the best scoring action and its evaluation are added to a list.
# When this process terminates, the list will indicate which action at the entry point is optimal.
# Further runs of this process will split the action spaces into increasingly smaller regions.
