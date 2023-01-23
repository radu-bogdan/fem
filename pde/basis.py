import numpy as np

def basis():
    BASIS = {}
    BASIS['RT0'] = {}
    BASIS['EJ1'] = {}
    BASIS['BDM1'] = {}
    BASIS['BDFM1'] = {}
    BASIS['RT1'] = {}
    BASIS['P0'] = {}
    BASIS['P1'] = {}
    BASIS['P1d'] = {}
    BASIS['P2'] = {}
    BASIS['Q0'] = {}
    BASIS['Q1'] = {}
    BASIS['Q1d'] = {}
    BASIS['Q2'] = {}

    #############################################################################################################
    # RT0 trig
    #############################################################################################################
    BASIS['RT0']['TRIG'] = {}
    BASIS['RT0']['TRIG']['phi'] = {}
    BASIS['RT0']['TRIG']['divphi'] = {}

    BASIS['RT0']['TRIG']['phi'][0] = lambda x,y: np.r_[x,y]
    BASIS['RT0']['TRIG']['phi'][1] = lambda x,y: np.r_[x-1,y]
    BASIS['RT0']['TRIG']['phi'][2] = lambda x,y: np.r_[x,y-1]

    BASIS['RT0']['TRIG']['divphi'][0] = lambda x,y: 2+0*x
    BASIS['RT0']['TRIG']['divphi'][1] = lambda x,y: 2+0*x
    BASIS['RT0']['TRIG']['divphi'][2] = lambda x,y: 2+0*x
    #############################################################################################################


    #############################################################################################################
    # RT0 quad
    #############################################################################################################
    BASIS['RT0']['QUAD'] = {}
    BASIS['RT0']['QUAD']['phi'] = {}
    BASIS['RT0']['QUAD']['divphi'] = {}

    BASIS['RT0']['QUAD']['phi'][0] = lambda x,y: np.r_[x,0*y]
    BASIS['RT0']['QUAD']['phi'][1] = lambda x,y: np.r_[0*x,y]
    BASIS['RT0']['QUAD']['phi'][2] = lambda x,y: np.r_[x-1,0*y]
    BASIS['RT0']['QUAD']['phi'][3] = lambda x,y: np.r_[0*x,y-1]

    BASIS['RT0']['QUAD']['divphi'][0] = lambda x,y: 1+0*x
    BASIS['RT0']['QUAD']['divphi'][1] = lambda x,y: 1+0*x
    BASIS['RT0']['QUAD']['divphi'][2] = lambda x,y: 1+0*x
    BASIS['RT0']['QUAD']['divphi'][3] = lambda x,y: 1+0*x
    #############################################################################################################


    #############################################################################################################
    # EJ1 trig
    #############################################################################################################
    BASIS['EJ1']['TRIG'] = {}
    BASIS['EJ1']['TRIG']['phi'] = {}
    BASIS['EJ1']['TRIG']['divphi'] = {}

    BASIS['EJ1']['TRIG']['phi'][3] = lambda x,y: np.r_[4*x*y,-4*x*y]
    BASIS['EJ1']['TRIG']['phi'][4] = lambda x,y: np.r_[0*x,4*y*(1-x-y)]
    BASIS['EJ1']['TRIG']['phi'][5] = lambda x,y: np.r_[4*x*(1-x-y),0*y]

    BASIS['EJ1']['TRIG']['divphi'][3] = lambda x,y: 4*y-4*x
    BASIS['EJ1']['TRIG']['divphi'][4] = lambda x,y: 4-4*x-8*y
    BASIS['EJ1']['TRIG']['divphi'][5] = lambda x,y: 4-8*x-4*y

    BASIS['EJ1']['TRIG']['phi'][0] = lambda x,y: BASIS['RT0']['TRIG']['phi'][0](x,y) -1/2*BASIS['EJ1']['TRIG']['phi'][4](x,y) -1/2*BASIS['EJ1']['TRIG']['phi'][5](x,y)
    BASIS['EJ1']['TRIG']['phi'][1] = lambda x,y: BASIS['RT0']['TRIG']['phi'][1](x,y) +1/2*BASIS['EJ1']['TRIG']['phi'][5](x,y) +1/2*BASIS['EJ1']['TRIG']['phi'][3](x,y)
    BASIS['EJ1']['TRIG']['phi'][2] = lambda x,y: BASIS['RT0']['TRIG']['phi'][2](x,y) -1/2*BASIS['EJ1']['TRIG']['phi'][3](x,y) +1/2*BASIS['EJ1']['TRIG']['phi'][4](x,y)

    BASIS['EJ1']['TRIG']['divphi'][0] = lambda x,y: BASIS['RT0']['TRIG']['divphi'][0](x,y) -1/2*BASIS['EJ1']['TRIG']['divphi'][4](x,y) -1/2*BASIS['EJ1']['TRIG']['divphi'][5](x,y)
    BASIS['EJ1']['TRIG']['divphi'][1] = lambda x,y: BASIS['RT0']['TRIG']['divphi'][1](x,y) +1/2*BASIS['EJ1']['TRIG']['divphi'][5](x,y) +1/2*BASIS['EJ1']['TRIG']['divphi'][3](x,y)
    BASIS['EJ1']['TRIG']['divphi'][2] = lambda x,y: BASIS['RT0']['TRIG']['divphi'][2](x,y) -1/2*BASIS['EJ1']['TRIG']['divphi'][3](x,y) +1/2*BASIS['EJ1']['TRIG']['divphi'][4](x,y)
    #############################################################################################################


    #############################################################################################################
    # BDM1 trig
    #############################################################################################################
    BASIS['BDM1']['TRIG'] = {}
    BASIS['BDM1']['TRIG']['phi'] = {}
    BASIS['BDM1']['TRIG']['divphi'] = {}

    BASIS['BDM1']['TRIG']['phi'][0] = lambda x,y: np.r_[x,0*y]
    BASIS['BDM1']['TRIG']['phi'][1] = lambda x,y: np.r_[0*x,y]
    BASIS['BDM1']['TRIG']['phi'][2] = lambda x,y: np.r_[-y,y]
    BASIS['BDM1']['TRIG']['phi'][3] = lambda x,y: np.r_[x+y-1,0*y]
    BASIS['BDM1']['TRIG']['phi'][4] = lambda x,y: np.r_[0*x,x+y-1]
    BASIS['BDM1']['TRIG']['phi'][5] = lambda x,y: np.r_[x,-x]

    BASIS['BDM1']['TRIG']['divphi'][0] = lambda x,y: 1+0*x
    BASIS['BDM1']['TRIG']['divphi'][1] = lambda x,y: 1+0*x
    BASIS['BDM1']['TRIG']['divphi'][2] = lambda x,y: 1+0*x
    BASIS['BDM1']['TRIG']['divphi'][3] = lambda x,y: 1+0*x
    BASIS['BDM1']['TRIG']['divphi'][4] = lambda x,y: 1+0*x
    BASIS['BDM1']['TRIG']['divphi'][5] = lambda x,y: 1+0*x
    #############################################################################################################


    #############################################################################################################
    # BDM1 quad
    #############################################################################################################
    BASIS['BDM1']['QUAD'] = {}
    BASIS['BDM1']['QUAD']['phi'] = {}
    BASIS['BDM1']['QUAD']['divphi'] = {}

    BASIS['BDM1']['QUAD']['phi'][0] = lambda x,y: 1/2*np.r_[-2*x*y+2*x,y**2-y]
    BASIS['BDM1']['QUAD']['phi'][1] = lambda x,y: 1/2*np.r_[2*x*y,-y**2+y]
    BASIS['BDM1']['QUAD']['phi'][2] = lambda x,y: 1/2*np.r_[-x**2+x,2*x*y]
    BASIS['BDM1']['QUAD']['phi'][3] = lambda x,y: 1/2*np.r_[x**2-x,2*y-2*x*y]
    BASIS['BDM1']['QUAD']['phi'][4] = lambda x,y: 1/2*np.r_[2*x*y-2*y,-y**2+y]
    BASIS['BDM1']['QUAD']['phi'][5] = lambda x,y: 1/2*np.r_[2*x+2*y-2-2*x*y,y**2-y]
    BASIS['BDM1']['QUAD']['phi'][6] = lambda x,y: 1/2*np.r_[x**2-x,2*x+2*y-2-2*x*y]
    BASIS['BDM1']['QUAD']['phi'][7] = lambda x,y: 1/2*np.r_[-x**2+x,2*x*y-2*x]

    BASIS['BDM1']['QUAD']['divphi'][0] = lambda x,y: 1/2+0*x
    BASIS['BDM1']['QUAD']['divphi'][1] = lambda x,y: 1/2+0*x
    BASIS['BDM1']['QUAD']['divphi'][2] = lambda x,y: 1/2+0*x
    BASIS['BDM1']['QUAD']['divphi'][3] = lambda x,y: 1/2+0*x
    BASIS['BDM1']['QUAD']['divphi'][4] = lambda x,y: 1/2+0*x
    BASIS['BDM1']['QUAD']['divphi'][5] = lambda x,y: 1/2+0*x
    BASIS['BDM1']['QUAD']['divphi'][6] = lambda x,y: 1/2+0*x
    BASIS['BDM1']['QUAD']['divphi'][7] = lambda x,y: 1/2+0*x
    #############################################################################################################


    #############################################################################################################
    # RT1 trig
    #############################################################################################################
    BASIS['RT1']['TRIG'] = {}
    BASIS['RT1']['TRIG']['phi'] = {}
    BASIS['RT1']['TRIG']['divphi'] = {}

    BASIS['RT1']['TRIG']['phi'][6] = lambda x,y: np.r_[x*y,y*(y-1)]
    BASIS['RT1']['TRIG']['phi'][7] = lambda x,y: np.r_[x*(x-1),x*y]

    BASIS['RT1']['TRIG']['divphi'][6] = lambda x,y: 3*y-1
    BASIS['RT1']['TRIG']['divphi'][7] = lambda x,y: 3*x-1

    BASIS['RT1']['TRIG']['phi'][0] = lambda x,y: BASIS['BDM1']['TRIG']['phi'][0](x,y) +1*BASIS['RT1']['TRIG']['phi'][6](x,y) +2*BASIS['RT1']['TRIG']['phi'][7](x,y)
    BASIS['RT1']['TRIG']['phi'][1] = lambda x,y: BASIS['BDM1']['TRIG']['phi'][1](x,y) +2*BASIS['RT1']['TRIG']['phi'][6](x,y) +1*BASIS['RT1']['TRIG']['phi'][7](x,y)
    BASIS['RT1']['TRIG']['phi'][2] = lambda x,y: BASIS['BDM1']['TRIG']['phi'][2](x,y) +1*BASIS['RT1']['TRIG']['phi'][6](x,y) -1*BASIS['RT1']['TRIG']['phi'][7](x,y)
    BASIS['RT1']['TRIG']['phi'][3] = lambda x,y: BASIS['BDM1']['TRIG']['phi'][3](x,y) -1*BASIS['RT1']['TRIG']['phi'][6](x,y) -2*BASIS['RT1']['TRIG']['phi'][7](x,y)
    BASIS['RT1']['TRIG']['phi'][4] = lambda x,y: BASIS['BDM1']['TRIG']['phi'][4](x,y) -2*BASIS['RT1']['TRIG']['phi'][6](x,y) -1*BASIS['RT1']['TRIG']['phi'][7](x,y)
    BASIS['RT1']['TRIG']['phi'][5] = lambda x,y: BASIS['BDM1']['TRIG']['phi'][5](x,y) -1*BASIS['RT1']['TRIG']['phi'][6](x,y) +1*BASIS['RT1']['TRIG']['phi'][7](x,y)

    BASIS['RT1']['TRIG']['divphi'][0] = lambda x,y: BASIS['BDM1']['TRIG']['divphi'][0](x,y) +1*BASIS['RT1']['TRIG']['divphi'][6](x,y) +2*BASIS['RT1']['TRIG']['divphi'][7](x,y)
    BASIS['RT1']['TRIG']['divphi'][1] = lambda x,y: BASIS['BDM1']['TRIG']['divphi'][1](x,y) +2*BASIS['RT1']['TRIG']['divphi'][6](x,y) +1*BASIS['RT1']['TRIG']['divphi'][7](x,y)
    BASIS['RT1']['TRIG']['divphi'][2] = lambda x,y: BASIS['BDM1']['TRIG']['divphi'][2](x,y) +1*BASIS['RT1']['TRIG']['divphi'][6](x,y) -1*BASIS['RT1']['TRIG']['divphi'][7](x,y)
    BASIS['RT1']['TRIG']['divphi'][3] = lambda x,y: BASIS['BDM1']['TRIG']['divphi'][3](x,y) -1*BASIS['RT1']['TRIG']['divphi'][6](x,y) -2*BASIS['RT1']['TRIG']['divphi'][7](x,y)
    BASIS['RT1']['TRIG']['divphi'][4] = lambda x,y: BASIS['BDM1']['TRIG']['divphi'][4](x,y) -2*BASIS['RT1']['TRIG']['divphi'][6](x,y) -1*BASIS['RT1']['TRIG']['divphi'][7](x,y)
    BASIS['RT1']['TRIG']['divphi'][5] = lambda x,y: BASIS['BDM1']['TRIG']['divphi'][5](x,y) -1*BASIS['RT1']['TRIG']['divphi'][6](x,y) +1*BASIS['RT1']['TRIG']['divphi'][7](x,y)
    #############################################################################################################


    #############################################################################################################
    # RT1 quad
    #############################################################################################################
    BASIS['BDFM1']['QUAD'] = {}
    BASIS['BDFM1']['QUAD']['phi'] = {}
    BASIS['BDFM1']['QUAD']['divphi'] = {}

    BASIS['BDFM1']['QUAD']['phi'][8] = lambda x,y: 1/2*np.r_[x*(x-1),0*x]
    BASIS['BDFM1']['QUAD']['phi'][9] = lambda x,y: 1/2*np.r_[0*y,y*(y-1)]

    BASIS['BDFM1']['QUAD']['divphi'][8] = lambda x,y: 1/2*(2*x-1)
    BASIS['BDFM1']['QUAD']['divphi'][9] = lambda x,y: 1/2*(2*y-1)

    BASIS['BDFM1']['QUAD']['phi'][0] = lambda x,y: BASIS['BDM1']['QUAD']['phi'][0](x,y) +2*BASIS['BDFM1']['QUAD']['phi'][8](x,y) -1*BASIS['BDFM1']['QUAD']['phi'][9](x,y)
    BASIS['BDFM1']['QUAD']['phi'][1] = lambda x,y: BASIS['BDM1']['QUAD']['phi'][1](x,y) +2*BASIS['BDFM1']['QUAD']['phi'][8](x,y) +1*BASIS['BDFM1']['QUAD']['phi'][9](x,y)
    BASIS['BDFM1']['QUAD']['phi'][2] = lambda x,y: BASIS['BDM1']['QUAD']['phi'][2](x,y) +1*BASIS['BDFM1']['QUAD']['phi'][8](x,y) +2*BASIS['BDFM1']['QUAD']['phi'][9](x,y)
    BASIS['BDFM1']['QUAD']['phi'][3] = lambda x,y: BASIS['BDM1']['QUAD']['phi'][3](x,y) -1*BASIS['BDFM1']['QUAD']['phi'][8](x,y) +2*BASIS['BDFM1']['QUAD']['phi'][9](x,y)
    BASIS['BDFM1']['QUAD']['phi'][4] = lambda x,y: BASIS['BDM1']['QUAD']['phi'][4](x,y) -2*BASIS['BDFM1']['QUAD']['phi'][8](x,y) +1*BASIS['BDFM1']['QUAD']['phi'][9](x,y)
    BASIS['BDFM1']['QUAD']['phi'][5] = lambda x,y: BASIS['BDM1']['QUAD']['phi'][5](x,y) -2*BASIS['BDFM1']['QUAD']['phi'][8](x,y) -1*BASIS['BDFM1']['QUAD']['phi'][9](x,y)
    BASIS['BDFM1']['QUAD']['phi'][6] = lambda x,y: BASIS['BDM1']['QUAD']['phi'][6](x,y) -1*BASIS['BDFM1']['QUAD']['phi'][8](x,y) -2*BASIS['BDFM1']['QUAD']['phi'][9](x,y)
    BASIS['BDFM1']['QUAD']['phi'][7] = lambda x,y: BASIS['BDM1']['QUAD']['phi'][7](x,y) +1*BASIS['BDFM1']['QUAD']['phi'][8](x,y) -2*BASIS['BDFM1']['QUAD']['phi'][9](x,y)

    BASIS['BDFM1']['QUAD']['divphi'][0] = lambda x,y: BASIS['BDM1']['QUAD']['divphi'][0](x,y) +2*BASIS['BDFM1']['QUAD']['divphi'][8](x,y) -1*BASIS['BDFM1']['QUAD']['divphi'][9](x,y)
    BASIS['BDFM1']['QUAD']['divphi'][1] = lambda x,y: BASIS['BDM1']['QUAD']['divphi'][1](x,y) +2*BASIS['BDFM1']['QUAD']['divphi'][8](x,y) +1*BASIS['BDFM1']['QUAD']['divphi'][9](x,y)
    BASIS['BDFM1']['QUAD']['divphi'][2] = lambda x,y: BASIS['BDM1']['QUAD']['divphi'][2](x,y) +1*BASIS['BDFM1']['QUAD']['divphi'][8](x,y) +2*BASIS['BDFM1']['QUAD']['divphi'][9](x,y)
    BASIS['BDFM1']['QUAD']['divphi'][3] = lambda x,y: BASIS['BDM1']['QUAD']['divphi'][3](x,y) -1*BASIS['BDFM1']['QUAD']['divphi'][8](x,y) +2*BASIS['BDFM1']['QUAD']['divphi'][9](x,y)
    BASIS['BDFM1']['QUAD']['divphi'][4] = lambda x,y: BASIS['BDM1']['QUAD']['divphi'][4](x,y) -2*BASIS['BDFM1']['QUAD']['divphi'][8](x,y) +1*BASIS['BDFM1']['QUAD']['divphi'][9](x,y)
    BASIS['BDFM1']['QUAD']['divphi'][5] = lambda x,y: BASIS['BDM1']['QUAD']['divphi'][5](x,y) -2*BASIS['BDFM1']['QUAD']['divphi'][8](x,y) -1*BASIS['BDFM1']['QUAD']['divphi'][9](x,y)
    BASIS['BDFM1']['QUAD']['divphi'][6] = lambda x,y: BASIS['BDM1']['QUAD']['divphi'][6](x,y) -1*BASIS['BDFM1']['QUAD']['divphi'][8](x,y) -2*BASIS['BDFM1']['QUAD']['divphi'][9](x,y)
    BASIS['BDFM1']['QUAD']['divphi'][7] = lambda x,y: BASIS['BDM1']['QUAD']['divphi'][7](x,y) +1*BASIS['BDFM1']['QUAD']['divphi'][8](x,y) -2*BASIS['BDFM1']['QUAD']['divphi'][9](x,y)
    #############################################################################################################


    #############################################################################################################
    # Pk
    #############################################################################################################
   
    BASIS['P0']['TRIG'] = {}
    BASIS['P0']['TRIG']['phi'] = {}
    BASIS['P0']['TRIG']['phi'][0] = lambda x,y: 1+0*x*y

    BASIS['P1']['TRIG'] = {}
    BASIS['P1']['TRIG']['phi'] = {}
    BASIS['P1']['TRIG']['phi'][0] = lambda x,y: 1-x-y
    BASIS['P1']['TRIG']['phi'][1] = lambda x,y: x
    BASIS['P1']['TRIG']['phi'][2] = lambda x,y: y
    
    BASIS['P1']['TRIG']['dphi'] = {}
    BASIS['P1']['TRIG']['dphi'][0] = lambda x,y: np.r_[-1,-1]
    BASIS['P1']['TRIG']['dphi'][1] = lambda x,y: np.r_[ 1, 0]
    BASIS['P1']['TRIG']['dphi'][2] = lambda x,y: np.r_[ 0, 1]

    BASIS['P1d']['TRIG'] = {}
    BASIS['P1d']['TRIG']['phi'] = {}
    BASIS['P1d']['TRIG']['phi'][0] = lambda x,y: 1-x-y
    BASIS['P1d']['TRIG']['phi'][1] = lambda x,y: x
    BASIS['P1d']['TRIG']['phi'][2] = lambda x,y: y
    
    BASIS['P1d']['TRIG']['dphi'] = {}
    BASIS['P1d']['TRIG']['dphi'][0] = lambda x,y: np.r_[-1,-1]
    BASIS['P1d']['TRIG']['dphi'][1] = lambda x,y: np.r_[ 1, 0]
    BASIS['P1d']['TRIG']['dphi'][2] = lambda x,y: np.r_[ 0, 1]

    BASIS['P2']['TRIG'] = {}
    BASIS['P2']['TRIG']['phi'] = {}
    BASIS['P2']['TRIG']['phi'][0] = lambda x,y: (1-x-y)*(1-2*x-2*y)
    BASIS['P2']['TRIG']['phi'][1] = lambda x,y: x*(2*x-1)
    BASIS['P2']['TRIG']['phi'][2] = lambda x,y: y*(2*y-1)
    BASIS['P2']['TRIG']['phi'][3] = lambda x,y: 4*x*(1-x-y)
    BASIS['P2']['TRIG']['phi'][4] = lambda x,y: 4*y*(1-x-y)
    BASIS['P2']['TRIG']['phi'][5] = lambda x,y: 4*x*y
    
    BASIS['P2']['TRIG']['dphi'] = {}
    BASIS['P2']['TRIG']['dphi'][0] = lambda x,y: np.r_[4*x+4*y-3, 4*x+4*y-3]
    BASIS['P2']['TRIG']['dphi'][1] = lambda x,y: np.r_[4*x-1, 0*x]
    BASIS['P2']['TRIG']['dphi'][2] = lambda x,y: np.r_[0*x, 4*y-1]
    BASIS['P2']['TRIG']['dphi'][3] = lambda x,y: np.r_[-4*(2*x+y-1), -4*x]
    BASIS['P2']['TRIG']['dphi'][4] = lambda x,y: np.r_[-4*y, -4*(x+2*y-1)]
    BASIS['P2']['TRIG']['dphi'][5] = lambda x,y: np.r_[4*y, 4*x]
    
    # BASIS.P2.phi{1}  = @(x,y) (1-x-y)*(1-2*x-2*y);
    # BASIS.P2.phi{2}  = @(x,y) x*(2*x-1);
    # BASIS.P2.phi{3}  = @(x,y) y*(2*y-1);
    # BASIS.P2.phi{4}  = @(x,y) 4*x*(1-x-y);
    # BASIS.P2.phi{5}  = @(x,y) 4*y*(1-x-y);
    # BASIS.P2.phi{6}  = @(x,y) 4*x*y;
    
    # BASIS.P2.dphi{1} = @(x,y) [4*x+4*y-3;4*x+4*y-3];
    # BASIS.P2.dphi{2} = @(x,y) [4*x-1;0];
    # BASIS.P2.dphi{3} = @(x,y) [0;4*y-1];
    # BASIS.P2.dphi{4} = @(x,y) [-4*(2*x+y-1); -4*x];
    # BASIS.P2.dphi{5} = @(x,y) [-4*y; -4*(x+2*y-1)];
    # BASIS.P2.dphi{6} = @(x,y) [4*y; 4*x];
    
    
    BASIS['Q0']['QUAD'] = {}
    BASIS['Q0']['QUAD']['phi'] = {}
    BASIS['Q0']['QUAD']['phi'][0] = lambda x,y: 1+0*x*y

    BASIS['P1']['QUAD'] = {}
    BASIS['P1']['QUAD']['phi'] = {}
    BASIS['P1']['QUAD']['phi'][0] = lambda x,y: 1-x-y
    BASIS['P1']['QUAD']['phi'][1] = lambda x,y: x
    BASIS['P1']['QUAD']['phi'][2] = lambda x,y: y

    BASIS['Q1']['QUAD'] = {}
    BASIS['Q1']['QUAD']['phi'] = {}
    BASIS['Q1']['QUAD']['phi'][0] = lambda x,y: (1-x)*(1-y)
    BASIS['Q1']['QUAD']['phi'][1] = lambda x,y: x*(1-y)
    BASIS['Q1']['QUAD']['phi'][2] = lambda x,y: x*y
    BASIS['Q1']['QUAD']['phi'][3] = lambda x,y: (1-x)*y

    BASIS['Q1']['QUAD']['dphi'] = {}
    BASIS['Q1']['QUAD']['dphi'][0] = lambda x,y: np.r_[y-1,x-1]
    BASIS['Q1']['QUAD']['dphi'][1] = lambda x,y: np.r_[1-y,-x]
    BASIS['Q1']['QUAD']['dphi'][2] = lambda x,y: np.r_[y,x]
    BASIS['Q1']['QUAD']['dphi'][3] = lambda x,y: np.r_[-y,1-x]

    BASIS['Q1d']['QUAD'] = {}
    BASIS['Q1d']['QUAD']['phi'] = {}
    BASIS['Q1d']['QUAD']['phi'][0] = lambda x,y: (1-x)*(1-y)
    BASIS['Q1d']['QUAD']['phi'][1] = lambda x,y: x*(1-y)
    BASIS['Q1d']['QUAD']['phi'][2] = lambda x,y: x*y
    BASIS['Q1d']['QUAD']['phi'][3] = lambda x,y: (1-x)*y

    BASIS['Q1d']['QUAD']['dphi'] = {}
    BASIS['Q1d']['QUAD']['dphi'][0] = lambda x,y: np.r_[y-1,x-1]
    BASIS['Q1d']['QUAD']['dphi'][1] = lambda x,y: np.r_[1-y,-x]
    BASIS['Q1d']['QUAD']['dphi'][2] = lambda x,y: np.r_[y,x]
    BASIS['Q1d']['QUAD']['dphi'][3] = lambda x,y: np.r_[-y,1-x]

    BASIS['Q2']['QUAD'] = {}
    BASIS['Q2']['QUAD']['phi'] = {}
    BASIS['Q2']['QUAD']['phi'][4] = lambda x,y: 4*x*y*(1-y)
    BASIS['Q2']['QUAD']['phi'][5] = lambda x,y: 4*x*y*(1-x)
    BASIS['Q2']['QUAD']['phi'][6] = lambda x,y: 4*y*(1-x)*(1-y)
    BASIS['Q2']['QUAD']['phi'][7] = lambda x,y: 4*x*(1-x)*(1-y)

    BASIS['Q2']['QUAD']['phi'][0] = lambda x,y: BASIS['Q1']['QUAD']['phi'][0] -1/2*BASIS['Q2']['QUAD']['phi'][6](x,y) -1/2*BASIS['Q2']['QUAD']['phi'][7]
    BASIS['Q2']['QUAD']['phi'][1] = lambda x,y: BASIS['Q1']['QUAD']['phi'][1] -1/2*BASIS['Q2']['QUAD']['phi'][4](x,y) -1/2*BASIS['Q2']['QUAD']['phi'][7]
    BASIS['Q2']['QUAD']['phi'][2] = lambda x,y: BASIS['Q1']['QUAD']['phi'][2] -1/2*BASIS['Q2']['QUAD']['phi'][4](x,y) -1/2*BASIS['Q2']['QUAD']['phi'][5]
    BASIS['Q2']['QUAD']['phi'][3] = lambda x,y: BASIS['Q1']['QUAD']['phi'][3] -1/2*BASIS['Q2']['QUAD']['phi'][5](x,y) -1/2*BASIS['Q2']['QUAD']['phi'][6]
    
    
    BASIS['P1']['B'] = {}
    BASIS['P1']['B']['phi'] = {}
    BASIS['P1']['B']['phi'][0] = lambda x: 1-x
    BASIS['P1']['B']['phi'][1] = lambda x: x
    
    BASIS['P2']['B'] = {}
    BASIS['P2']['B']['phi'] = {}
    BASIS['P2']['B']['phi'][0] = lambda x: (1-x)*(1-2*x)
    BASIS['P2']['B']['phi'][1] = lambda x: x*(2*x-1)
    BASIS['P2']['B']['phi'][2] = lambda x: 4*x*(1-x)
    
    # DUAL BASIS (DOFs)
    
    BASIS['DOF'] = {}
    BASIS['DOF']['HDIV'] = {}
    
    BASIS['DOF']['HDIV']['RT0'] = {}  
    BASIS['DOF']['HDIV']['RT0'][0] = lambda x: 1+0*x
    
    BASIS['DOF']['HDIV']['BDM1'] = {}
    BASIS['DOF']['HDIV']['BDM1'][0] = lambda x: 6*x-2 # dual to x and 1-x on the edge...
    BASIS['DOF']['HDIV']['BDM1'][1] = lambda x: -6*x+4
    
    
    
    return BASIS






if __name__ == "__main__":
    BASIS = basis()







