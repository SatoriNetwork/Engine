"""
option 1: cache at the engine or even streamodel level

for each stream, store all observations as they are, add more as they come in
pos: smallest, simplest 
con: duplication at the interface level

option 2: cache at the interface level

the data from history, modified and conformed, etc, combined with others, etc.
pos: no duplication, so it's the most efficient, and fastest
con: complicated, incrementally complicated

we will implement option 1 first. maybe option 2 later.
"""
