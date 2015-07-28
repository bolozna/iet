"""Module for analysing event-based data.

Definition of some concepts used in this module:

edge : Edge in a temporal network. A tuple (i,j,t) containing two node indices i and j and a time stamp t.
event : An event taking place at some time t. Usually represented by the time stamp t.
event sequence : A sorted sequence of events. Usually represented by sorted list of time stamps.
inter-event time : Time between two subsequent events.
IET : inter-event time
IET sequence : sequence of inter-event times. Usually represented as a list of IETs.

"""

import scipy.stats
import scipy.optimize
from scipy.special import binom
import random
import math
import itertools


class CumDist(object):
    """Cumulative probability distribution.
    """
    def __init__(self,vals,ps,maxval=None):
        self.vals=vals
        self.ps=ps
        self.maxval=maxval

    def rescale(self,newavg):
        """Returns a rescaled distribution

        Parameters
        ----------
        newavg : float 
           The average of the new distribution  
        """
        new_vals=[]
        c=newavg/float(self.get_moment(1))
        for val in self.vals:
            new_vals.append(c*val)
        return CumDist(new_vals,copy(self.ps))

    def get_moment(self,moment):
        s=0
        for i in range(1,len(self.ps)):
           s+=(self.ps[i-1]-self.ps[i])*(self.vals[i]**moment)
        if self.ps[-1]!=0:
            s+=self.ps[-1]*(self.maxval**moment)
        return s


class KaplanMeierEstimator(object):
    def __init__(self,censored_times=None,event_times=None):
        """Create a new instance of the KM estimator. 

        Parameters
        ----------
        censored_times : Dict or None
           Initial number of censored times given as dictionary with keys indicating censored times and values indicating the numbers of times each censored time is observed. If None, then no censored times are added to the estimator object initially.
        event_times : Dict or None
           Initial number of event times given as dictionary with keys indicating event times and values indicating the numbers of times each event time is observed. If None, then no event times are added to the estimator object initially.
        """
        if censored_times==None:
            self.censor_times={}
        else:
            self.censor_times=censored_times
        if event_times==None:
            self.event_times={}
        else:
            self.event_times=event_times

    def add_event(self, time, count=1):        
        self.event_times[time]=self.event_times.get(time,0)+count

    def add_events(self,times):
        for time in times:
            self.add_event(time)

    def add_censored(self, time, count=1):
        self.censor_times[time]=self.censor_times.get(time,0)+count

    def add_censoreds(self,times):
        for time in times:
            self.add_censored(time)

    def get_cumulative(self):
        t_set=set(self.event_times.keys()).union(set(self.censor_times.keys()))
        t_list=sorted(t_set)
        del t_set # just to save some memory

        events_cum,events_sum=[],0
        censored_cum,censored_sum=[],0
        for t in t_list:
            events_sum+=self.event_times.get(t,0)
            events_cum.append(events_sum)
            censored_sum+=self.censor_times.get(t,0)
            censored_cum.append(censored_sum)
        return t_list,events_cum,censored_cum
        

    def get_estimator(self,variance=False,correctionFactor=1.0):
        """Returns the Kaplan-Meier estimator.

        Parameters
        ----------
        variance : Boolean
           Controls the return values (see Returns)

        Returns
        -------
        If 'variance' if false a tuple (t,s) is returned, and
        if 'variance' if true a tuple (t,s,v) is returned.

        t : an ordered array of times
        s : the estimator value at each time
        v : Variance of the estimator (Greenwood's formula)
        """
        t_list,events_cum,censored_cum=self.get_cumulative()

        ndts=events_cum[-1]+censored_cum[-1]
        Si=[1]
        Ti=[0]

        Vmod=0
        Vi=[1*Vmod]

        for i,t in enumerate(t_list):
            if i==0:
                ni=ndts
            else:
                ni=ndts-events_cum[i-1]-censored_cum[i-1]
            di=self.event_times.get(t,0)
            if ni>0:
                if di>0:
                    Ti.append(t)
                    si=(ni-di)/float(ni)
                    Si_prev= Si[-1] if len(Si)>0 else 1
                    Si.append(Si_prev*si)
                    if (ni-di)!=0:
                        Vmod+=di/float(ni*(ni-di))
                        Vi.append(Si[-1]**2 * Vmod)
                    else:
                        Vi.append(None)

        if variance:
            return Ti,Si,Vi
        else:
            return Ti,Si

    @staticmethod
    def get_confidence_intervals(Si,Vi,confLevel,logcorrect=False,correctionFactor=1.0):
        """Returns confidence intervals for the KM estimator.
        """
        assert len(Si)==len(Vi)
        Si_up=[1]
        Si_down=[1]
        z=scipy.stats.norm.ppf(1-(1.-confLevel)/2.)
        if logcorrect:
            def loginv(w):
                return math.exp(w)/float(1+math.exp(w))
            for i in range(1,len(Si)):
                w=math.log(Si[i]/(1.0-Si[i]))
                wvar=(1./float(Si[i]*(1-Si[i])))**2*correctionFactor*Vi[i]
                Si_up.append(loginv(w+z*math.sqrt(wvar)))
                Si_down.append(loginv(w-z*math.sqrt(wvar)))
        else:
            for i in range(1,len(Si)):
                Si_up.append(Si[i]+z*math.sqrt(correctionFactor*Vi[i]))
                Si_down.append(Si[i]-z*math.sqrt(correctionFactor*Vi[i]))
        return Si_up,Si_down


    def get_naive_estimator(self):
        """Returns an estimator where the censored events are simply discarded.
        """
        t_list,events_cum,censored_cum=self.get_cumulative()

        ndts=events_cum[-1]
        Si=[1]
        Ti=[0]
        for i,t in enumerate(t_list):
            if i==0:
                ni=ndts
            else:
                ni=ndts-events_cum[i-1]
            di=self.event_times.get(t,0)
            if ni>0:
                if di>0:
                    Ti.append(t)
                    si=(ni-di)/float(ni)
                    Si_prev= Si[-1] if len(Si)>0 else 1
                    Si.append(Si_prev*si)                

        return Ti,Si

class IntereventTimeEstimator(object):
    def __init__(self,endTime,mode='censorlast'):
        """ Constructor.
        
        Parameters
        ----------
        endTime : float
           The last time point in the observation period
        mode : string
           'censorlast' : the last iet is censored when the observation period ends
           'censorall' : in addition to the last iet, the first one is censored
           'periodic' : periodic boundary conditions
        
        """
        self.endTime=endTime
        assert mode in ["censorlast","censorall","periodic"]
        self.mode=mode

        self.observed_iets={}
        self.forward_censored_iets={}
        self.backward_censored_iets={}
        self.empty_seqs=0
        self.nseqs=0

    def add_time_seq(self,seq):
        self.nseqs+=1
        if len(seq)!=0:
            for i,time in enumerate(seq):
                if i!=0:
                    dt=time-last
                    assert dt>=0
                    self.observed_iets[dt]=self.observed_iets.get(dt,0)+1
                elif self.mode=='censorall':
                    self.backward_censored_iets[time]=self.backward_censored_iets.get(time,0)+1
                elif self.mode=='periodic':
                    firstTime=time
                last=time
            if self.mode=='periodic':
                dt=firstTime+self.endTime-last
                self.observed_iets[dt]=self.observed_iets.get(dt,0)+1
            else:
                dt=self.endTime-last
                self.forward_censored_iets[dt]=self.forward_censored_iets.get(dt,0)+1
        else:
            self.empty_seqs+=1

    def read_seqs(self,filename):
        """Reads a file containing event sequences.
        """
        f=open(filename,'r')
        for line in f:
            seq=map(float,line.split())
            self.add_time_seq(seq)

    def get_estimator(self,variance=False):
        """Returns the KM estimator for the IETs.
        """

        if self.mode=='periodic':
            censored_times=None
            event_times=self.observed_iets
        elif self.mode=='censorlast':
            censored_times=self.forward_censored_iets
            event_times=self.observed_iets
        elif self.mode=='censorall':
            censored_times=self.backward_censored_iets.copy()
            for key,val in self.forward_censored_iets.iteritems():
                censored_times[key]=censored_times.get(key,0)+val
            event_times=self.observed_iets.copy()
            for key,val in self.observed_iets.iteritems():
                event_times[key]=event_times[key]*2

        km_estimator=KaplanMeierEstimator(censored_times=censored_times,event_times=event_times)
        return km_estimator.get_estimator(variance=variance)

    def get_naive_estimator(self):
        """Returns the distribution of observed IETs.
        """
        km_estimator=KaplanMeierEstimator(event_times=self.observed_iets)
        return km_estimator.get_naive_estimator()

    def get_npmle_estimator(self,return_mu=False):
        """Returns a nonparametric maximum likelihood estimator for IETs assuming they are produced with a stationary renewal process.

        This implementes the modified RT algorithm in G. Soon and M. Woodroofe "Nonparametric estimation and consistency for renewal processes", Journal of Statistical Planning and Inference 53 (1996) pp. 171--195.
        """
        assert self.mode=="censorall"
        #setup
        nx=sum(self.observed_iets.itervalues())
        ny=sum(self.backward_censored_iets.itervalues())
        nz=sum(self.forward_censored_iets.itervalues())
        nw=self.empty_seqs
        ts=list(set(itertools.chain(self.observed_iets.iterkeys(),self.forward_censored_iets.iterkeys(),self.backward_censored_iets.iterkeys())))
        if nw!=0:
            ts.append(self.endTime)
        ts.sort()
        
        #step a
        p_old=[1./float(len(ts)) for key in ts]
        v_old=nw/float(ny+nw)

        while True:
            #step b
            r=[]
            sumi=0
            sum_p_old=sum(map(lambda j:p_old[j],range(len(ts))))
            for k in range(len(ts)-1):
                t=ts[k]
                sum_p_old-=p_old[k-1] if k>0 else 0.
                sumi=sumi+(self.forward_censored_iets.get(t,0)+self.backward_censored_iets.get(t,0))/float(sum_p_old)

                r.append(self.observed_iets.get(t,0)+p_old[k]*sumi)
                
            r.append(p_old[-1]*sumi)

            if v_old!=0:
                rh1=v_old*nw/float(v_old )
            else:
                rh1=0

            f=lambda mu:sum( (r[k]*ts[k]/float((nx+nz)*mu+(ny+nw)*ts[k]) for k in range(len(ts))) )-1+rh1/float(ny+nw)
            mu_new=scipy.optimize.bisect(f,0,100*ts[-1])
            
            #step c
            p_new=[]
            for k,t in enumerate(ts):
                p_new.append(r[k]*mu_new/float((nx+nz)*mu_new+(ny+nw)*t ))
            v_new=rh1*mu_new/float(ny+nw)

            #step d
            if sum(map(lambda x,y:abs(x-y),p_old,p_new))>10**-4:
                p_old=p_new
                v_old=v_new
            else:
                cump=[1]
                if ts[0]!=0:
                    ts.insert(0,0)
                for pval in p_new:
                    cump.append(cump[-1]-pval)
                if return_mu:
                    return ts,cump,mu_new
                else:
                    return ts,cump


    def estimate_moment(self,moment,method="naive"):
        """Returns an estimate for a moment of the inter-event time distribution.

        Choose one of the following methods.
        'naive' : Moment of the observed inter-event time distribution
        'km' : Moment of a distribution estimated using the KM estimator
        """ 
        s,n=0,0
        if method=="naive":
            for iet,num in self.observed_iets.iteritems():
                s+=num*(iet**moment)
                n+=num
            if n!=0:
                return s/float(n)
            else:
                return None
        elif method=="km":
            niets=sum(self.observed_iets.itervalues())
            if niets>1 :
                ts,ps=self.get_estimator()
                cd=CumDist(ts,ps,maxval=ts[-1])
                return cd.get_moment(moment)
            elif niets==0:
                return self.endTime**moment
            elif niets==1:
                return (self.endTime/2.)**moment

        else:
            raise Exception("Invalid parameter value for 'method': "+method)


def edges_to_timeseqs(edges, issorted=True):
    """Generator transforming edges to time sequences.
    """
    if not issorted:
        edges=sorted(edges)
    current=None
    for event in edges:
        if (event[0],event[1]) != current:
            if current!=None:
                l.sort()
                yield l
            l=[]
        l.append(event[2])
        current=(event[0],event[1])
    l.sort()
    yield l


def iets(events):
    """Generator for inter-event times.
    """
    for i,event in enumerate(events):
        if i!=0:
            yield event-lastevent
        lastevent=event

def normalize(events,form="timeseqs"):
    """Normalizes times in an event list in place.

    Normalization is done such that the event times are between 0 and 1. For 
    edge the network is in addition made undirected in a way that the smaller
    node in the edge is always given before the larger one. E.g., (2,1,t) is
    transformed into (1,2,t).
    """
    mint,maxt=None,None

    if form=="edges":
        for fr,to,t in events:
            if maxt==None or t>maxt:
                maxt=t
            if mint==None or t<mint:
                mint=t
        for i in range(len(events)):
            events[i][2]=events[i][2]-mint
            if events[i][0]>events[i][1]:
                temp=events[i][0]
                events[i][0]=events[i][1]
                events[i][1]=temp
    if form=="timeseqs":
        for timeseq in events:
            for t in timeseq:
                if maxt==None or t>maxt:
                    maxt=t
                if mint==None or t<mint:
                    mint=t
        for timeseq in events:
            for i,t in enumerate(timeseq):
                timeseq[i]=timeseq[i]-mint

    return maxt-mint


def generate_renewal_process(tdist,trdist,endtime,starttime=0):
    """ Generate a time sequence using a stationary renewal process.

    Parameters
    ----------
    tdist : function
      A function returning realizations from the inter-event time distribution
    trdist : function
      A function returning realizations from the residual waiting time distribution
    endtime : float
      The end time of the observation period
    starttime : float
      The start time of the observation period

    Returns
    -------
    A sequence of times of the events.
    """
    l=[]
    t1=trdist()
    if t1+starttime>endtime:
        return l
    else:
        l.append(t1+starttime)
    while True:
        ti=tdist()
        if l[-1]+ti>endtime:
            break
        l.append(l[-1]+ti)

    return l

def generate_renewal_process_burnin(tdist,endtime,starttime=0,burninfactor=10):
    """ Generate a time sequence using a stationary renewal process when the residual waiting time distribution is not specified.

    The stationarity is ensured only approximately by using a burn-in time. The process is first simulated starting from non-stationary state for the duration of the burn-in time, and the start time is set to the end of the burn-in time.

    Parameters
    ----------
    tdist : function
      A function returning realizations from the inter-event time distribution
    endtime : float
      The end time of the observation period
    starttime : float
      The start time of the observation period
    burninfactor : float
      The burn-in time is burninfactor*(endtime-starttime).
    

    Returns
    -------
    A sequence of times of the events.
    """

    t=0
    dt=endtime-starttime
    bt=dt*burninfactor
    l=[]
    while True:
        ti=tdist()
        t=t+ti
        if t>bt:
            if starttime+t-bt > endtime:
                return l
            l.append(starttime+t-bt)
            break
    while True:
        ti=tdist()
        if l[-1]+ti>endtime:
            break
        l.append(l[-1]+ti)

    return l



def exprv(rate):
    """Generate a realization of an exponentially distributed random variable.
    """
    p=random.random()
    return -math.log(p)/float(rate)

def generate_renewal_process_exp(rate,starttime,endtime):
    """Generate a time sequence using a stationary Poisson process (i.e., exponential IET distribution).
    """
    return generate_renewal_process(lambda :exprv(rate),lambda :exprv(rate),endtime,starttime)

def plaw(exp,mint=1.):
    """Generate a value from power-law distribution.

    Probability density is:
    p(\tau) = 0, when \tau < \tau_m
    p(\tau) = (\alpha - 1) \tau_m^{\alpha - 1} \tau^{-\alpha}, \tau > \tau_m    
    """
    p=random.random()
    return mint * (1. - p)**(1./float(1.-exp))

def plaw_residual(exp,mint=1.):
    """Generate a value from the distribution of the residual of power-law IET.

    Probability density is:
    p(\tau_R) = \frac{\alpha - 2}{\alpha - 1} \tau_m^{-1}, when \tau_R < \tau_m
    and
    p(\tau_R) = \frac{\alpha - 2}{\alpha - 1} \tau_m^{\alpha - 2} * \tau_R^{1-\alpha}, when \tau_R > \tau_m

    Cumulative probability distribution is:
    P(\tau_R) = \frac{\alpha - 2}{\alpha -1} \tau_m^{-1} \tau_R, when \tau_R < \tau_m
    and
    P(\tau_R) = 1 - \frac{\tau_m^{\alpha - 2}}{\alpha - 1} \tau_R^{2 - \alpha}, when \tau_R > \tau_m
    
    Inverse cumulative probability distribution:
    P^{-1}(p) = \frac{\alpha - 1}{\alpha - 2} \tau_m p, when p < \frac{\alpha - 2}{\alpha - 1}
    and
    P^{-1}(p) = \tau_m ((\alpha - 1)(1-p))^{frac{1}{2 - \alpha}}, when p < \frac{\alpha - 2}{\alpha - 1}
    """
    p=random.random()
    if p < float(exp - 2.)/float(exp - 1.):
        return float(exp-1.)/float(exp-2.)*mint*p
    else:
        return mint*((exp -1.)*(1.-p))**(1./float(2.-exp))

def generate_renewal_process_plaw(exp,mint,endtime,starttime=0,burnin=None):
    """Generate a time sequence using a stationary renewal process with power-law IET distribution
    """
    if burnin==None:
        return generate_renewal_process(lambda :plaw(exp,mint),lambda :plaw_residual(exp,mint),endtime,starttime)
    else:
        return generate_renewal_process_burnin(lambda :plaw(exp,mint),endtime,starttime,burninfactor=burnin)

def create_scaled_sequence(uldist,uldist_residual,avgdist,n,T):
    """Generates a time sequence using a stationary renewal process with IET distribution that is scaled.
    """
    for i in range(n):
        navg=avgdist()
        ndist=lambda :navg*uldist()
        ndist_residual=lambda :navg*uldist_residual()
        yield generate_renewal_process(ndist,ndist_residual,T,0)


if __name__=="__main__":
    km=KaplanMeierEstimator()
    km.add_event(3)
    km.add_event(11,2)
    km.add_censored(9)
    km.add_censored(12,6)
    t,s=km.get_estimator()
    print t
    print s


    tn,sn=km.get_naive_estimator()
    print tn
    print sn

    for mode in ['censorlast','censorall','periodic']:
        iet_est=IntereventTimeEstimator(12,mode=mode)
        iet_est.add_time_seq([3,4,5,9])
        print mode,
        print ": ",
        print iet_est.get_naive_estimator(),
        print ", ",
        print iet_est.get_estimator(variance=True)

    iet_est=IntereventTimeEstimator(12,mode='censorall')
    iet_est.add_time_seq([3,4,5,9])
    print iet_est.get_npmle_estimator()
