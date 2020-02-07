import timeit

time=timeit.default_timer

class timer:
    def __init__(self):
        self.elapsedTime=0
        self.startTime=0
        self.status="stopped"
        
    def start(self):
        self.startTime=time()
    def elapsed(self):
        if self.startTime==0:
            return self.elapsedTime
        else:
            return self.elapsedTime + time() - self.startTime
        
    def stop(self):
        diff=time() - self.startTime
        self.elapsedTime+=diff
        self.startTime=0
        return diff
        
    def toogle(self):
        if self.startTime==0:
            return self.start()
        else:
            return self.stop()
    def __repr__(self):
        if self.startTime==0:
            status="stopped"
        else:
            status="ticking"
        
        return "<elaps=" + str(self.elapsed() ) + "," + status + ">"
        

        
