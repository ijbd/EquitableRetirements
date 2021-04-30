from EquitableRetirement import EquitableRetirement
from CoalPlants import CoalPlants
from RenewableSites import RenewableSites

import numpy as np
import pandas as pd 

def test_large():
    ''' use sample data to test runtime and large-scale functionality of formulation '''
    print('TEST_LARGE:')
    print('\t','getting data...')

    ### GET DATA
    numYears = 10

        # coal plants
    plants = CoalPlants.getCoalPlants(['NY','PA','OH','WV','KY','TN','VA','MD','DE','NC','NJ'])
    plants['HISTGEN'] = CoalPlants.getPlantGeneration(plants['Plant Code'])
    plants['HD'] = CoalPlants.getMarginalHealthCosts(plants['Plant Code'])
    plants.dropna(inplace=True)

        # renewables
    reSites = RenewableSites.getAnnualCF()

        # costs
    costs = { 'RECAPEX' : np.append(np.ones(len(reSites)//2)*1600.,np.ones(len(reSites)//2)*1700.),
              'REFOPEX' : np.append(np.ones(len(reSites)//2)*19.,np.ones(len(reSites)//2)*43.),
              'COALVOPEX' : np.ones(len(plants))*(4.+11*2.2),
              'COALFOPEX' : np.ones(len(plants))*40.
            }

        # site limits
    limits = { 'MAXCAP' : np.ones((len(reSites),len(plants)))*1e3,
               'SITEMAXCAP' : np.ones(len(reSites))*1e3,
               'MAXSITES' : np.ones(len(plants))*10
             }
             
        # employment factors
    ef = { 'RETEF' : np.ones(len(plants))*.25,
           'CONEF' : np.ones((len(reSites),numYears))*.25,
           'COALOMEF' : np.ones(len(plants)),
           'REOMEF' : np.ones((len(reSites),numYears))*.25
        }

    ### BUILD MODEL
    m = EquitableRetirement()

    m.Y = np.arange(numYears)+2020
    m.R = reSites.index.values
    m.C = plants.index.values

    m.Params.HISTGEN = plants['HISTGEN'].values
    m.Params.COALCAP = plants['Coal Capacity (MW)'].values
    m.Params.CF = reSites['Annual CF'].values
    m.Params.RECAPEX = costs['RECAPEX']
    m.Params.REFOPEX = costs['REFOPEX']
    m.Params.COALVOPEX = costs['COALVOPEX']
    m.Params.COALFOPEX = costs['COALFOPEX']
    m.Params.MAXCAP = limits['MAXCAP']
    m.Params.SITEMAXCAP = limits['SITEMAXCAP']
    m.Params.MAXSITES = limits['MAXSITES']
    m.Params.HD = plants['HD'].values
    m.Params.RETEF = ef['RETEF']
    m.Params.CONEF = ef['CONEF']
    m.Params.COALOMEF = ef['COALOMEF']
    m.Params.REOMEF = ef['REOMEF']

    ### CHECK DIMS
    print('\t','Y\t',len(m.Y))
    print('\t','R\t',len(m.R))
    print('\t','C\t',len(m.C))
    print('\t','HISTGEN\t',m.Params.HISTGEN.shape)
    print('\t','COALCAP\t',m.Params.COALCAP.shape)
    print('\t','CF\t',m.Params.CF.shape)
    print('\t','RECAPEX\t',m.Params.RECAPEX.shape)
    print('\t','REFOPEX\t',m.Params.REFOPEX.shape)
    print('\t','COALVOPEX\t',m.Params.COALVOPEX.shape)
    print('\t','COALFOPEX\t',m.Params.COALFOPEX.shape)
    print('\t','MAXCAP\t',m.Params.MAXCAP.shape)
    print('\t','SITEMAXCAP\t',m.Params.SITEMAXCAP.shape)
    print('\t','MAXSITES\t',m.Params.MAXSITES.shape)
    print('\t','HD\t',m.Params.HD.shape)
    print('\t','RETEF\t',m.Params.RETEF.shape)
    print('\t','CONEF\t',m.Params.CONEF.shape)
    print('\t','COALOMEF\t',m.Params.COALOMEF.shape)
    print('\t','REOMEF\t',m.Params.REOMEF.shape)

    print('\t','')

    print('\t','solving...')

    m.solve(1,0,0)

    print('\t',m.Output.Z)

    pass

def test_cplex():
    ''' small test to see if the cplex is working '''
    print('TEST_CPLEX:')

    numYears = 10
    numCoal = 5
    numRE = 10

    ### BUILD MODEL
    m = EquitableRetirement()

    m.Y = np.arange(numYears)+2020
    m.R = np.arange(numRE)
    m.C = np.arange(numCoal)

    m.Params.HISTGEN = np.ones(numCoal)*8760
    m.Params.COALCAP = np.ones(numCoal)
    m.Params.CF = np.ones(numRE)*.75
    m.Params.RECAPEX = np.ones(numRE)
    m.Params.REFOPEX = np.ones(numRE)
    m.Params.COALVOPEX = np.ones(numCoal)
    m.Params.COALFOPEX = np.ones(numCoal)
    m.Params.MAXCAP = np.ones((numRE,numCoal))*10
    m.Params.SITEMAXCAP = np.ones(numRE)*10
    m.Params.MAXSITES = np.ones(numCoal)*10
    m.Params.HD = np.ones(numCoal)*0
    m.Params.RETEF = np.ones(numCoal)*0
    m.Params.CONEF = np.ones((numRE,numYears))*0
    m.Params.COALOMEF = np.ones(numCoal)*0
    m.Params.REOMEF = np.ones((numRE,numYears))*0

    ### CHECK DIMS
    print('\t','Y\t',len(m.Y))
    print('\t','R\t',len(m.R))
    print('\t','C\t',len(m.C))
    print('\t','HISTGEN\t',m.Params.HISTGEN.shape)
    print('\t','COALCAP\t',m.Params.COALCAP.shape)
    print('\t','CF\t',m.Params.CF.shape)
    print('\t','RECAPEX\t',m.Params.RECAPEX.shape)
    print('\t','REFOPEX\t',m.Params.REFOPEX.shape)
    print('\t','COALVOPEX\t',m.Params.COALVOPEX.shape)
    print('\t','COALFOPEX\t',m.Params.COALFOPEX.shape)
    print('\t','MAXCAP\t',m.Params.MAXCAP.shape)
    print('\t','SITEMAXCAP\t',m.Params.SITEMAXCAP.shape)
    print('\t','MAXSITES\t',m.Params.MAXSITES.shape)
    print('\t','HD\t',m.Params.HD.shape)
    print('\t','RETEF\t',m.Params.RETEF.shape)
    print('\t','CONEF\t',m.Params.CONEF.shape)
    print('\t','COALOMEF\t',m.Params.COALOMEF.shape)
    print('\t','REOMEF\t',m.Params.REOMEF.shape)

    print('\t','')

    print('\t','solving...')

    m.solve(1,0,0,'cplex')

    print('\t',m.Output.Z)

    pass


if __name__ == '__main__':
    #test_large()
    #test_cplex()
