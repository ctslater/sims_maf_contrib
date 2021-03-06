# Transient metric with input ascii lightcurve
# fbb@nyu.edu, svalenti@lcogt.net

import numpy as np

from lsst.sims.maf.metrics import BaseMetric

class TransientAscii(BaseMetric):
    """
    Written according to the transient metric example.
    Calculate what fraction of the transients would be detected. Best paired with a spatial slicer.
    The lightcurve in input is an ascii file per photometric band so that different lightcurve
    shapes can be implemented.
    It also allows a different detection threshold for each filter in units of 5sigma's
    """
    def __init__(self, metricName='TransientAsciiMetric', mjdCol='expMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter',
                 surveyDuration=10., surveyStart=None, detectM5Plus=0.,
                 detectfactor={'u':1,'g':1,'r':1,'i':1,'z':1,'y':1},
                 maxdiscT=5, nPreT=0,nPerLC=1, nFilters=1, nPhaseCheck = 1,
                 peakOffset=0,
                 asciifile={'u':'','g':'','r':'','i':'','z':'','y':''},
                 **kwargs):
        """
        transDuration = how long the transient lasts (days)
        peakOffset = magnitude offset compared to ascii file mag value (m_band+offset for each band).
        surveyDuration = Length of survey (years).
        surveyStart = MJD for the survey start date (otherwise us the time of the first observation).
        detectfactor = detection threshold per filter in units of 5 sigma's
        maxdiscT = latest time acceptable for first detection (discovery)
        nPreT = Number of observations (any filter(s)) to demand before Time maxdiscT
                  before saying a transient has been detected.
        nPerLC = Number of sections of the light curve that must be sampled above the detectM5Plus theshold
                 (in a single filter) for the light curve to be counted. For example,
                 setting nPerLC = 2 means a light curve  is only considered detected if there
                 is at least 1 observation in the first half of the LC,
                 and at least one in the second half of the LC. nPerLC = 4 means each quarter of the light curve
                 must be detected to count.
        nPhaseCheck = number of different phases that will be tested
        nFilters = Number of filters that need to be observed for an object to be counted as detected.
        """
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        super(TransientAscii, self).__init__(col=[self.mjdCol, self.m5Col,self.filterCol],
                                                 units='Fraction Detected',metricName=metricName,**kwargs)

        self.surveyDuration = surveyDuration
        self.surveyStart = surveyStart
        self.detectM5Plus = detectM5Plus
        self.detectfactor  = detectfactor
        self.maxdiscT = maxdiscT
        self.nPreT = nPreT
        self.peakOffset = peakOffset
        self.nPerLC = nPerLC
        self.nFilters = nFilters
        self.nPhaseCheck = nPhaseCheck
        self.asciifile = asciifile
        self.transDuration=0.0

    def read_lightCurve_SV(self):
        """
        reads in an ascii file, 3 columns: epoch, magnitude, filter
        """
        if not self.asciifile:
            return None
        else:
            data = np.genfromtxt(self.asciifile,dtype=[('ph','f8'),('mag','f8'), ('flt','S1')])
               
            self.transDuration=data['ph'].max()-data['ph'].min()
        return data

    def make_lightCurve_SV(self, lcv_dict, time, filters):
        
        lcMags = np.zeros(time.size, dtype=float)
        
        for key in set(lcv_dict['flt']):
                fMatch_ascii = (np.array(lcv_dict['flt']) == key)
                Lc_ascii_filter= np.interp(time,np.array(lcv_dict['ph'],float)[fMatch_ascii],
                                           np.array(lcv_dict['mag'],float)[fMatch_ascii])
                     
                #fMatch = np.where(filters == key)
                lcMags[filters==key]=Lc_ascii_filter[filters==key]
        lcMags+=self.peakOffset

   
        return lcMags

    def run(self, dataSlice, slicePoint=None):

        # Total number of transients that could go off back-to-back
        inlcv_dict = self.read_lightCurve_SV()
        nTransMax = np.floor(self.surveyDuration/(self.transDuration/365.25))
        tshifts = np.arange(self.nPhaseCheck)*self.transDuration/float(self.nPhaseCheck)
        nDetected = 0

        

        for tshift in tshifts:
            # Compute the total number of back-to-back transients are possible to detect
            # given the survey duration and the transient duration.
            nTransMax += np.floor(self.surveyDuration/(self.transDuration/365.25))
            if tshift != 0:
                nTransMax -= 1
            if self.surveyStart is None:
                surveyStart = dataSlice[self.mjdCol].min()
            time = (dataSlice[self.mjdCol] - surveyStart + tshift) % self.transDuration


            # Which lightcurve does each point belong to
            lcNumber = np.floor((dataSlice[self.mjdCol]-surveyStart)/self.transDuration)

            lcMags = self.make_lightCurve_SV(inlcv_dict, time, dataSlice[self.filterCol])
            

            # How many criteria needs to be passed
            detectThresh = 0

            # Flag points that are above the SNR limit
            detected = np.zeros(dataSlice.size, dtype=int)

            ###FBB modified to allow a different threshold for each filter
            #print dataSlice.dtype.names
            factor=np.array([self.detectfactor[f] for f in dataSlice[self.filterCol]])
            detected[np.where(lcMags < dataSlice[self.m5Col]-2.5*np.log10 (factor*np.sqrt(factor)))] += 1
            detectThresh += 1

            # If we demand points before a specified T (maxdectT)
            try: float(self.nPreT)
            except:
                preT=np.array([self.nPreT[i] for i in ['u','g','r','i','z','y']])
                if preT.any() > 0:
                    self.nPreT=1
            if self.nPreT>0:
                detectThresh += 1
                ord = np.argsort(dataSlice[self.mjdCol])
                dataSlice = dataSlice[ord]
                detected = detected[ord]
                lcNumber = lcNumber[ord]
                time = time[ord]
                ulcNumber = np.unique(lcNumber)
                left = np.searchsorted(lcNumber, ulcNumber)
                right = np.searchsorted(lcNumber, ulcNumber, side='right')

                for le,ri in zip(left,right):
                    # Number of points where there are a detection
                    good = np.where(time[le:ri] < self.maxdiscT)
                    nd = np.sum(detected[le:ri][good])
                    if nd >= self.nPreT:
                        detected[le:ri] += 1

            # Check if we need multiple points per light curve or multiple filters
            if (self.nPerLC > 1) | (self.nFilters > 1) :
                # make sure things are sorted by time
                ord = np.argsort(dataSlice[self.mjdCol])
                dataSlice = dataSlice[ord]
                detected = detected[ord]
                lcNumber = lcNumber[ord]
                ulcNumber = np.unique(lcNumber)
                left = np.searchsorted(lcNumber, ulcNumber)
                right = np.searchsorted(lcNumber, ulcNumber, side='right')
                detectThresh += self.nFilters

                for le,ri in zip(left,right):
                    points = np.where(detected[le:ri] > 0)
                    ufilters = np.unique(dataSlice[self.filterCol][le:ri][points])
                    phaseSections = np.floor(time[le:ri][points]/self.transDuration * self.nPerLC)
                    for filtName in ufilters:
                        good = np.where(dataSlice[self.filterCol][le:ri][points] == filtName)
                        if np.size(np.unique(phaseSections[good])) >= self.nPerLC:
                            detected[le:ri] += 1

            # Find the unique number of light curves that passed the required number of conditions
            nDetected += np.size(np.unique(lcNumber[np.where(detected >= detectThresh)]))

  

        return float(nDetected)/nTransMax

