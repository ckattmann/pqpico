import cProfile, pstats, StringIO

pr = cProfile.Profile()
pr.enable()
try:
    execfile('runPQ.py')
finally:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(0.01)
    print s.getvalue()
