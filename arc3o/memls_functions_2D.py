
# Created on Wed Aug 20 17:39 2018
#
# Created for the arc3o package
# This file contains the MEMLS functions
#
# @author: Clara Burgard, http://www.github.com/ClimateClara
# based on a 1D Matlab version developed by A. Wiesmann and C. Mätzler, and 
#		 extended for sea ice by R.T. Tonboe, see Wiesmann and Mätzler (1998), Wiesmann and Mätzler (1999)
#		 and Tonboe (2006)
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

#####################################################

import numpy as np
import xarray as xr
import arc3o.profile_functions as pf
import timeit

########################################################

def epsice(Ti,freq):

    """Compute dielectric permittivity of ice

    This function computes the dielectric permittivity of ice, after Hufford, Mitzima and Matzler.

    Parameters
    ----------
    Ti: np.array or xarray.DataArray
        ice temperature in K
    freq: float
        frequency in GHz

    Returns
    -------
    eice: np.array or xarray.DataArray
        dielectric permittivity of ice

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    pp = (300/Ti)-1;
    B1 = 0.0207
    b = 335.25
    B2 = 1.16e-11
    db = np.exp(-10.02+0.0364*(Ti-273.15)) #replaced 273 by 273.15 by Clara
    beta = ((B1*np.exp(b/Ti))/(Ti*(np.exp(b/Ti)-1)**2))+B2*freq**2+db
    alpha = ((0.00504 + 0.0062*pp)* np.exp(-22.1*pp))
    eice = (alpha/freq) + (beta*freq)

    return eice



def epsr(roi):

    """Compute real part of dielectric permittivity for dry snow

    This function computes the real part of dielectric permittivity for dry snow from density.

    Parameters
    ----------
    roi: np.array or xarray.DataArray
        snow density in g/cm3

    Returns
    -------
    epsi: np.array or xarray.DataArray
        real part of dielectric permittivity of dry snow

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    vfi = roi/0.917
    ehb = 0.99913
    esb = 1.4759

    epsi1 = 1 + 1.5995 * roi.where(roi<=0.4) + 1.861 * roi.where(roi<=0.4)**3
    epsi2 = ((1 - vfi.where(roi>0.4)) * ehb + vfi.where(roi>0.4) * esb)**3
    epsi = epsi1.where(roi<=0.4,0)+epsi2.where(roi>0.4,0)
    epsi = epsi.where(roi>0)

    return epsi


def ro2epsd(roi,Ti,freq):

    """Compute real part of dielectric permittivity for dry snow

    This function computes the dielectric permittivity for dry snow from density.

    Parameters
    ----------
    roi: np.array or xarray.DataArray
        snow density in g/cm3
    Ti: np.array or xarray.DataArray
        snow temperature in K
    freq: float
        frequency in GHz

    Returns
    -------
    epsi: np.array or xarray.DataArray
        real part of dielectric permittivity of dry snow
    epsii: np.array or xarray.DataArray
        imaginary part of dielectric permittivity of dry snow

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    eice = epsice(Ti,freq)
    epsi = epsr(roi)

    #imaginary part after Tiuri 84
    #epsii = eice.*(0.52.*roi + 0.62.*(roi.^2));

    #imaginary part after Polder and van Santen 1946 (Effective-Medium Approx)
    f = roi/0.917
    ei = 3.185
    #N = len(roi)
    A = roi*0 + 0.3
    #epsp = roi*0

    A1 = 0.476 - 0.64 * f.where(f<0.55)
    A2 = 0.1 + 0.5 * f.where(f<=0.333)
    A = A1.where(f<0.55,0) + A2.where(f<=0.333,0)
    A = A.where(f<0.55,0.3)

    epsp = epsi
    A3 = 1-2*A
    ea = (epsp*(1-A))+A
    ea3 = epsp*(1-A3)+A3
    K1 = (ea/(ea+A*(ei-1)))**2
    K3 = (ea3/(ea3+A3*(ei-1)))**2
    Ksq = (2*K1+K3)/3
    epsii = np.sqrt(epsi)*eice*Ksq*f
    return epsi, epsii


def mixmod(f,Ti,Wi,epsi,epsii):

    """Compute permittivity for snow wetness above 0

    This function computes the permittivity for Wetness > 0 using the physical mixing model Weise 97,
    after Matzler 1987 (corrected). The water temperature is assumed constant at 273.15 K.

    Parameters
    ----------
    f: float
        frequency in GHz
    Ti: np.array or xarray.DataArray
        snow/ice temperature in K
    Wi: np.array or xarray.DataArray
        snow/ice wetness between 0 and 1
    epsi: np.array or xarray.DataArray
        real part of dielectric permittivity of dry snow
    epsii: np.array or xarray.DataArray
        imaginary part of dielectric permittivity

    Returns
    -------
    epsi: np.array or xarray.DataArray
        real part of dielectric permittivity of wet snow
    epsii: np.array or xarray.DataArray
        imaginary part of dielectric permittivity of wet snow

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    Aa = 0.005
    Ab = 0.4975
    Ac = 0.4975
    euw = 4.9
    esw = 88.045
    frw = 0.11109 # inverse relaxation frequency of water

    esa = (esw - epsi)/(3*(1+Aa*(esw/epsi-1)))
    esb = (esw - epsi)/(3*(1+Ab*(esw/epsi-1)))
    esc = (esw - epsi)/(3*(1+Ac*(esw/epsi-1)))
    eua = (euw - epsi)/(3*(1+Aa*(euw/epsi-1)))
    eub = (euw - epsi)/(3*(1+Ab*(euw/epsi-1)))
    euc = (euw - epsi)/(3*(1+Ac*(euw/epsi-1)))

    fa = 1 + Aa * (esw-euw)/(epsi+Aa*(euw-epsi))
    fb = 1 + Ab * (esw-euw)/(epsi+Ab*(euw-epsi))
    fc = 1 + Ac * (esw-euw)/(epsi+Ac*(euw-epsi))

    eea = esa - eua
    eeb = esb - eub
    eec = esc - euc

    fwa = frw/fa
    fwb = frw/fb
    fwc = frw/fc

    depsia = eua + eea / (1+(fwa*f)**2)
    depsib = eub + eeb / (1+(fwb*f)**2)
    depsic = euc + eec / (1+(fwc*f)**2)
    depsi = Wi * (depsia+depsib+depsic)

    depsiia = fwa*f*eea / (1+(fwa*f)**2)
    depsiib = fwb*f*eeb / (1+(fwb*f)**2)
    depsiic = fwc*f*eec / (1+(fwc*f)**2)
    depsii = Wi * (depsiia + depsiib + depsiic)

    epsi = epsi + depsi
    epsii = epsii + depsii
    return epsi, epsii


def epice(T,freq):

    """Compute the dielectric constant of pure ice.

    This function computes the dielectric constant of pure ice, based on :cite:`matzler98`.

    Parameters
    ----------
    T: np.array or xarray.DataArray
        ice temperature in K or °C
    freq: float
        frequency in GHz

    Returns
    -------
    epui: np.array or xarray.DataArray
        dielectric constant of pure ice

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    if T.max() < 100:
          T = T+273.15
    epi=3.1884+9.1e-4*(T-273.15) #modified from 273 to 273.15 by Clara
    #The Hufford model for the imaginary part.
    theta=300/T-1 #corrected by Clara 09.05.2018
    alpha=(0.00504+0.0062*theta)*np.exp(-22.1*theta)
    beta=(0.502-0.131*theta/(1+theta))*1e-4 +(0.542e-6*((1+theta)/(theta+0.0073))**2)
    epii=(alpha/freq) + (beta*freq)
    epui=epi+epii*1j
    return epui

def Nsw(Ssw):

    """Compute normality of sea water or brine

    This function computes the normality of sea water or brine. It is Eq. 20 in :cite:`ulaby86`.

    Parameters
    ----------
    Ssw: np.array or xarray.DataArray
        salinity of brine or sea water in g/kg
    freq: float
        frequency in GHz

    Returns
    -------
    N: np.array or xarray.DataArray
        normality of sea water or brine

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    N = 0.9141*Ssw*(1.707e-2 +1.205e-5*Ssw+4.058e-9*(Ssw**2))
    return N


def condbrine(T):

    """Compute the conductivity of brine

    This function computes the conductivity of brine. It is Eq. 7 in :cite:`stogryn85`.

    Parameters
    ----------
    T: np.array or xarray.DataArray
        ice temperature in K or °C

    Returns
    -------
    condbrine: np.array or xarray.DataArray
        conductivity of brine

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    if T.max() > 100:
          T=T-273.15

    condbrine1 = -T.where(T>=-22.9)*np.exp(0.5193 + 0.8755*0.1*T.where(T >= -22.9))
    condbrine2 = -T.where(T<-22.9)*np.exp(1.0334 + 0.1100*T.where(T<-22.9))
    condbrine = condbrine1.where(T>=-22.9,0) + condbrine2.where(T<-22.9,0)
    condbrine = condbrine.where(T>-100)

    return condbrine


def relaxt(T):

    """Compute relaxation time

    This function computes the relaxation time, the fit is valid up to -25°C. It is Eq. 12 in :cite:`stogryn85`.

    Parameters
    ----------
    T: np.array or xarray.DataArray
        ice temperature in K or °C

    Returns
    -------
    relax: np.array or xarray.DataArray
        relaxation time in nanoseconds

    Notes
    -----
    This function was introduced into the MEMLS code by `R.T. Tonboe <http://research.dmi.dk/staff/all-staff/rtt/>`_.
    It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    if T.max() > 100:
        T=T-273.15
    relax = (1./(2*np.pi))*(0.10990 + 0.13603*0.01*T + 0.20894*0.001*T**2 + 0.28167*0.00001*T**3)
    t0 = (1./(2*np.pi))*0.1121
    relax = relax.where(T!=0,t0)
    relax = relax/1e9
    return relax


def epsib0(T):

    """Compute static dielectric constant of brine.

    This function computes the static dielectric constant of brine. The fit is valid up to -25°C. It is Eq. 10 in :cite:`stogryn85`.

    Parameters
    ----------
    T: np.array or xarray.DataArray
        ice temperature in K or °C

    Returns
    -------
    epsib0: np.array or xarray.DataArray
        static dielectric constant of brine

    Notes
    -----
    This function was introduced into the MEMLS code by `R.T. Tonboe <http://research.dmi.dk/staff/all-staff/rtt/>`_.
    It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    if T.max() > 100:
        T=T-273.15
    epsib0 = (939.66 - 19.068*T)/(10.737 - T)
    epsib0 = epsib0.where(T!=0,87.92)
    return epsib0



def ebrine(T,freq):

    """Compute brine permittivity

    This function computes the brine permittivity, the fit is valid up to -25°C. It is Eq. 1 in :cite:`stogryn85`.

    Parameters
    ----------
    T: np.array or xarray.DataArray
        ice temperature in K or °C
    freq: float
        frequency in GHz

    Returns
    -------
    ebr.real: np.array or xarray.DataArray
        real part of brine permittivity
    ebr.imag: np.array or xarray.DataArray
        imaginary part of brine permittivity

    Notes
    -----
    This function was introduced into the MEMLS code by `R.T. Tonboe <http://research.dmi.dk/staff/all-staff/rtt/>`_.
    It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    if T.max() > 100:
        T=T-273.15

    f = freq*1e9
    e0 = 8.85419*1e-12

    epsiwoo = (82.79 + 8.19*T**2)/(15.68 + T**2)
    epsiwoo = epsiwoo.where(T!=0,5.28)

    epsis = epsib0(T)
    tau = relaxt(T)
    sig = condbrine(T)

    ebr = epsiwoo + (epsis - epsiwoo)/(1.-2*np.pi*f*tau*1j) + (1j*sig)/(2*np.pi*e0*f)

    return ebr.real, ebr.imag



def eice_s2p(e1,e2,v):

    """Compute effective dielectric constant of medium consisting of e1 and e2

    This function computes the effective dielectric constant of medium consisting of e1 and e2. It is based on the
    improved born approximation by :cite:`matzler98b` and the Polder/VanSanten mixing formulae for spherical inclusions.

    Parameters
    ----------
    e1: np.array or xarray.DataArray
        dielectric constant of background
    e2: np.array or xarray.DataArray
        dielectric constant of sherical inclusions
    v: np.array or xarray.DataArray
        fraction of inclusions

    Returns
    -------
    eeff: np.array or xarray.DataArray
        effective dielectric constant of medium consisting of e1 and e2

    Notes
    -----
    This function was introduced into the MEMLS code by `R.T. Tonboe <http://research.dmi.dk/staff/all-staff/rtt/>`_.
    It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    eeff=0.25*(2.*e1-e2+3.*v*(e2-e1)+np.sqrt((2.*e1-e2+3.*v*(e2-e1))**2 +8.*e1*e2))
    return eeff



def sie(si,sal,Ti,freq,epsi,epsii):

    """Compute dielectric constant of ice if it is an ice layer

    This function computes the dielectric constant of ice if it is a sea-ice layer. Formulae from :cite:`ulaby86`.

    Parameters
    ----------
    si: np.array or xarray.DataArray
        sea ice/snow layer 1 or 0
    sal: np.array or xarray.DataArray
        salinity in g/kg
    Ti: np.array or xarray.DataArray
        ice temperature in K
    freq: float
        frequency in GHz
    epsi: np.array or xarray.DataArray
        initial permittivity (of snow)
    epsii: np.array or xarray.DataArray
        initial loss (of snow)

    Returns
    -------
    epsi: np.array or xarray.DataArray
        permittivity of ice
    epsii: np.array or xarray.DataArray
        loss of ice

    Notes
    -----
    This function was introduced into the MEMLS code by `R.T. Tonboe <http://research.dmi.dk/staff/all-staff/rtt/>`_.
    It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    if Ti.max() > 100:
        Ti=Ti-273.15 #get therm. temp. in C
    eice=epice(Ti,freq)                  #fresh ice dielectric constant
    Sbr = pf.Sb(Ti)                      #added by Clara due to the new form of the Vb function
    volb=pf.Vb(sal,Sbr)                  #volume of brine
    [eb,ebi]=ebrine(Ti,freq)             #dielectric constant of brine
    #emis=eice_rn2p(eice,eb+ebi*i,volb)  #dielectric constant of sea ice (random needles)
    emis=eice_s2p(eice,eb+ebi*1j,volb)    #dielectric constant of sea ice (spherical inclusions)

    aepsi=emis.real
    aepsii=emis.imag
    ###added by Clara
    #aepsii[aepsii>1.] = 1.
    ###

    epsi=epsi-epsi*si+aepsi*si
    epsii=epsii-epsii*si+aepsii*si
    return epsi, epsii



def mysie(si,rho,Ti,sal,freq,epsi,epsii):
    """Compute dielectric constant of ice if it is a mutliyear ice layer

    This function computes the dielectric constant of ice if it is a multiyear ice layer. Formulae from :cite:`ulaby86`.

    Parameters
    ----------
    si: np.array or xarray.DataArray
        sea ice/snow layer 1 or 0
    rho: np.array or xarray.DataArray
        density of ice layer in g/cm3
    Ti: np.array or xarray.DataArray
        ice temperature in K
    sal: np.array or xarray.DataArray
        salinity in g/kg
    freq: float
        frequency in GHz
    epsi: np.array or xarray.DataArray
        initial permittivity (of snow)
    epsii: np.array or xarray.DataArray
        initial loss (of snow)

    Returns
    -------
    epsi: np.array or xarray.DataArray
        permittivity of ice
    epsii: np.array or xarray.DataArray
        loss of ice

    Notes
    -----
    This function was introduced into the MEMLS code by `R.T. Tonboe <http://research.dmi.dk/staff/all-staff/rtt/>`_.
    It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    #permittivity of saline ice
    [sepsi, sepsii] = sie(si,sal,Ti,freq,epsi,epsii)

    eice=epice(Ti,freq)                           #fresh ice dielectric constant
    vola=(0.926 - rho)/0.926                      #volume of air
    ###modified by Clara
    vola = vola.where(vola>=0,0)
    ###
    emis=eice_s2p(sepsi + 1j*sepsii,1.0 + 0.0j,vola)   #dielectric constant of sea ice (spherical inclusions)

    aepsi=emis.real
    aepsii=emis.imag

    epsi=epsi-epsi*si+aepsi*si
    epsii=epsii-epsii*si+aepsii*si

    ##added by Clara
    #epsii[epsii>1] = 1.
    ###

    return epsi, epsii




def abscoeff(epsi,epsii,Ti,freq):
    """Compute absorption coefficient.

    This function computes the absorption coefficient from the dielectric properties. Formulae from :cite:`ulaby86`.

    Parameters
    ----------
    epsi: np.array or xarray.DataArray
        real part dielectric constant
    epsii: np.array or xarray.DataArray
        imaginary part dielectric constant
    Ti: np.array or xarray.DataArray
        ice temperature in K
    freq: float
        frequency in GHz

    Returns
    -------
    gai: np.array or xarray.DataArray
        absorption coefficient

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    # constants
    c = 2.99793

    lamd=c/(10*freq)
    gai=(4.0*np.pi/lamd)*(np.sqrt(epsi+epsii*1j)).imag
    # Absorption coefficient, suitable for snow but not saline ice > about 10psu
  	#gai = ((2*pi*10*freq).*epsii)./(c.*sqrt(epsi - (epsii.^2./4.*epsi)));
    return gai



def tei_ndim(teta,ns):
    """Append ``teta`` at the end of the dimension ``layer_nb``

    This function has practical reasons, it appends ``teta`` at the end of ``ns`` over the dimension ``layer_nb``

    Parameters
    ----------
    teta: float
        incidence angle in degrees
    ns: xarray.DataArray
        real part of the refractive index of the slab

    Returns
    -------
    tei_ndim: xarray.DataArray
        local incidence angle

    Notes
    -----
    This function was introduced by `C. Burgard <http://www.github.com/ClimateClara>`_ to adapt the python functions to
    :py:class:`xarray.DataArray` parameters and output.
    """

    layer_max = ns['layer_nb'].max()
    teta0 = ns.sel(layer_nb=layer_max)*0 + teta
    ns2 = np.arcsin(np.sin(teta)/ns)
    tei0 =  xr.concat([ns2, teta0], dim='layer_nb')
    return tei0


def append_laydim_end(xrda,const):

    """Append a constant at the end of the dimension ``layer_nb``

    This function has practical reasons, it appends a constant at the end of a dimension, here ``layer_nb``

    Parameters
    ----------
    xrda: xarray.DataArray
        variable where it has to be appended
    const: float
        constant to be appended at the end

    Returns
    -------
    xrda_app: xarray.DataArray
        longer array

    Notes
    -----
    This function was introduced by `C. Burgard <http://www.github.com/ClimateClara>`_ to adapt the python functions to
    :py:class:`xarray.DataArray` parameters and output.
    """

    const_ndim = xrda.isel(layer_nb=0)*0 + const
    xrda_app = xr.concat([xrda, const_ndim], dim='layer_nb')
    return xrda_app


def append_laydim_begin(xrda,const):
    """Append a constant at the beginning of the dimension ``layer_nb``

    This function has practical reasons, it appends a constant at the beginning of a dimension, here ``layer_nb``.

    Parameters
    ----------
    xrda: xarray.DataArray
        variable where it has to be appended
    const: float
        constant to be appended at the beginning

    Returns
    -------
    xrda_app: xarray.DataArray
        longer array

    Notes
    -----
    This function was introduced by `C. Burgard <http://www.github.com/ClimateClara>`_ to adapt the python functions to
    :py:class:`xarray.DataArray` parameters and output.
    """

    const_ndim = xrda.isel(layer_nb=0)*0 + const
    xrda_app = xr.concat([const_ndim, xrda], dim='layer_nb')
    return xrda_app



def pfadi(tei,di):
    """Compute effective path length in a layer

    This function computes the effective path length in a layer.

    Parameters
    ----------
    tei: xarray.DataArray
        local incidence angle
    di: xarray.DataArray
        ice thickness of layer in m

    Returns
    -------
    dei: xarray.DataArray
        effective path length in m

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    N = len(di.layer_nb)
    dei = di/np.cos(tei.isel(layer_nb=range(0,N)))
    return dei


def fresnelc0(tei,epsi):
    """Compute the fresnel reflection coefficients

    This function computes the fresnel reflection coefficients (assuming eps" = 0), layer n+1 is the air above the snowpack

    Parameters
    ----------
    tei: xarray.DataArray
        local incidence angle
    epsi: xarray.DataArray
        real part of dielectric permittivity

    Returns
    -------
    sih: xarray.DataArray
        interface reflectivity at h pol
    siv: xarray.DataArray
        interface reflectivity at v pol

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    N = len(epsi.layer_nb)-1
    siv = epsi.isel(layer_nb=range(N))*0
    sih = epsi.isel(layer_nb=range(N))*0

    for n in range(N):
        epso = epsi.isel(layer_nb=n+1)
        epsu = epsi.isel(layer_nb=n)
        tein = tei.isel(layer_nb=n+1)
        #in the following loc is n+1 because we are not at isel but at sel
        sih.loc[dict(layer_nb=n+1)] = ((np.sqrt(epso)*np.cos(tein) - np.sqrt(epsu - epso * np.sin(tein)**2))/(np.sqrt(epso)*np.cos(tein) + np.sqrt(epsu - epso * np.sin(tein)**2)))**2
        siv.loc[dict(layer_nb=n+1)] = ((epsu*np.cos(tein) - np.sqrt(epso)*np.sqrt(epsu - epso*np.sin(tein)**2))/(epsu*np.cos(tein) + np.sqrt(epso)*np.sqrt(epsu - epso*np.sin(tein)**2)))**2

    return sih, siv



def sccoeff(roi,Ti,pci,freq,Wi,gai,sccho):
    """Compute the the scattering coefficient

    This function computes the the scattering coefficient from structural parameters. Different algorithms can be chosen,
    by changing "sccho"

    Parameters
    ----------
    roi: np.array or xarray.DataArray
        density in g/cm3
    Ti: np.array or xarray.DataArray
        temperature in K
    pci: np.array or xarray.DataArray
        correlation length in mm
    freq: float
        frequency in GHz
    Wi: np.array or xarray.DataArray
        wetness between 0 and 1
    gai: np.array or xarray.DataArray
        absorption coefficient
    sccho: int
        scattering coefficient algorithm chosen

    Returns
    -------
    gbih: np.array or xarray.DataArray
        2-flux scattering coefficient at h pol
    gbiv: np.array or xarray.DataArray
        2-flux scattering coefficient at v pol
    gs6: np.array or xarray.DataArray
        6-flux scattering coefficient
    ga2i: np.array or xarray.DataArray
        2-flux absorption coefficient

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`.     It was translated to Python and adapted for multi-dimensional input by
    `C. Burgard <http://www.github.com/ClimateClara>`_ to be used in ARC3O.
    """

    ## constants
    c = 2.99
    roair = 0.001293
    roice = 0.917
    ## specular component of scattering coefficient
    ## usually 0 can be important in new snow!
    dgb0h = 0
    dgb0v = 0
    ## aus der Theorie scattering coefficient
    k = freq*(2*np.pi/0.299793)
    eice = 3.18
    vfi = roi/roice

    ## choose the scattering algorithm that should be used
    wahl = sccho

    [epsi,epsii] = ro2epsd(roi,Ti,freq)
    [epsi,epsii] = mixmod(freq,Ti,Wi,epsi,epsii)

    ## 6-flux scattering coefficient
    if wahl == 1:
        gs6 = ((130 * ((freq/50)**2.7)) * pci**3) / (roi**1.3 + 0.001)


    ## fit vom 26.8.97 auf alle Daten v-pol, > 11 GHz
    if wahl == 2:
        gs6 = 0.0704 * (pci**2.32)*(freq**1.68)*roi**(-0.63)


    ## for spheres: Mätzler, J. Appl. Phys. 83(11) 6111-6117 eqs 27+32 iborn
    epseff = (2.-eice+3.*vfi*(eice-1)+ np.sqrt((2.-eice+3.*vfi*(eice-1))**2+8.*eice))/4.
    sphe = (3./32)*(0.001*pci)**3*k**4*vfi*(1-vfi)*abs((2.*epseff+1)*(eice-1)/(2.*epseff+eice))**2
    if wahl == 4:
        gs6 = sphe

    ## for shells(new and recrystalized snow): 
    ## Mätzler, J. Appl. Phys. 83(11) 6111-6117 eq 39 iborn
    epseff = 1.+(vfi*(eice-1)*(2.+1/eice))/(3.-vfi*(1-1./eice))
    shel = abs(2./3 + 1./(3.*eice**2))*(0.001*pci)*k**2*vfi*(1-vfi)*(eice-1)**2./(16.*epseff)
    if wahl == 5:
        gs6 = shel

    ## as linearcombination
    if wahl == 6:
        a = 0.1664
        b = 0.2545
        gs6 = a*sphe+b*shel

    ## fit vom 26.9.97
    if wahl == 7:
        gs6 = 73.21 * (pci**3)*((freq/50)**2.68)*roi**(-1)

    ## fit vom 13.10.97
    if wahl == 8:
        gs6 = 136 * (pci**2.85) * ((freq/50)**2.5) / (roi + 0.001)

    ## fit vom 4.11.97 (without density)
    if wahl == 9:
        gs6 = 564 * (pci**3.0)* ((freq/50)**2.5)

    ## fit vom 4.11.97 (without density, uses corr. length from exp. fit!)
    if wahl == 10:
        gs6 = (3.16 * pci + 295 * (pci**2.5))* ((freq/50)**2.5)

    ## fit vom 4.11.97 (with density, uses corr. length from exp. fit!)
    if wahl == 11:
        gs6 = (9.20 * pci - 1.23 * roi + 0.54)**2.5 * ((freq/50)**2.5)

    omega = np.sqrt((epsi - 1.)/epsi)


    ## Born Approximation
    if wahl == 12:
        print('Born approximation, missing the functions!')
    #    kp = bornsnk(roi,0)
    #    [gb6,gc6,gf6,gs6] = borna(k,vfi,pci,epsi,eice,epseff,kp)
    else:
        gb6 = 0.5 * gs6 * (1.-omega)
        gc6 = 0.25 * gs6 * omega

    ## -> 2 Flux
    gtr = (4. * gc6) / (gai + 2. * gc6) #gc is coefficient for coupling between horiz and vert fluxes
    ga2i = gai * (1. + gtr) #two-flux absorption coefficient

    gbih = (gb6 + dgb0h) + gtr * gc6
    gbiv = (gb6 + dgb0v) + gtr * gc6

    return gbih,gbiv,gs6,ga2i



def meteo_sc(si,rroi,rTi,rpci,freq,rWi,rgai,gbih,gbiv,gs6,ga2i):
    """Compute the scattering coefficient of fresh snow

    This function computes the scattering coefficient of only partly recrystallized snow, linear combination of iborn, wahl==6 (see :func:`sccoeff`).

    Parameters
    ----------
    si: xarray.DataArray
        layer type ice/snow [1/0]
    rroi: xarray.DataArray
        density in g/cm3
    rTi: xarray.DataArray
        temperature in K or °C
    rpci: xarray.DataArray
        correlation length in mm
    freq: float
        frequency in GHz
    rgai: xarray.DataArray
        absorption coefficient
    gbih: xarray.DataArray
        2-flux scattering coefficient at h pol
    gbiv: xarray.DataArray
        2-flux scattering coefficient at v pol
    gs6: xarray.DataArray
        6-flux scattering coefficient
    ga2i: xarray.DataArray
        2-flux absorption coefficient

    Returns
    -------
    gbih: xarray.DataArray
        2-flux scattering coefficient at h pol
    gbiv: xarray.DataArray
        2-flux scattering coefficient at v pol
    gs6: xarray.DataArray
        6-flux scattering coefficient
    ga2i: xarray.DataArray
        2-flux absorption coefficient

    Notes
    -----
    This function was introduced into the MEMLS code by `R.T. Tonboe <http://research.dmi.dk/staff/all-staff/rtt/>`_.
    It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    [dumgbih,dumgbiv,ags6,dumga2i] = sccoeff(rroi,rTi,rpci,freq,rWi,rgai,6)
    gs6=gs6-gs6.where(si==1,0)*1+ags6.where(si==1,0)*1
    return gbih,gbiv,gs6,ga2i



def iborn_s2p(e1,e2,eeff,v,k,pcc):
    """Compute the scattering coefficient with improved born approximation

    This function computes the scattering coefficient of a collection of spherical inclusions with correlation length pcc using improved born approximation
    (see :cite:`matzler98b`).

    Parameters
    ----------
    e1: np.array or xarray.DataArray
        dielectric constant of background
    e2: np.array or xarray.DataArray
        dielectric constant of sherical inclusions
    eeff: np.array or xarray.DataArray
        effective dielectric constant of medium consisting of e1 and e2
    v: np.array or xarray.DataArray
        brine volume fraction
    k: float
        given constant (f(freq))
    pcc: np.array or xarray.DataArray
        correlation length in mm

    Returns
    -------
    ss:	np.array or xarray.DataArray
        6-flux scattering coefficient for ice

    Notes
    -----
    This function was introduced into the MEMLS code by `R.T. Tonboe <http://research.dmi.dk/staff/all-staff/rtt/>`_.
    It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    ss=(3. *pcc**3 *k**4 /32.) *v *(1-v) *abs(((e2-e1)*(2. *eeff+e1)) /(2. *eeff+e2))**2
    return ss




def scice(si,gbih,gbiv,gs6,ga2i,Ti,sal,freq,pci):
    """Compute the ice scattering coefficient from structural parameters

    This function computes the scattering coefficient from structural parameters

    Parameters
    ----------
    si: np.array or xarray.DataArray
        layer type ice/snow [1/0]
    gbih: np.array or xarray.DataArray
        2-flux scattering coefficient at h pol
    gbiv: np.array or xarray.DataArray
        2-flux scattering coefficient at v pol
    gs6: np.array or xarray.DataArray
        6-flux scattering coefficient
    ga2i: np.array or xarray.DataArray
        2-flux absorption coefficient
    Ti:	xarray.DataArray
        temperature in K
    sal: np.array or xarray.DataArray
        salinity in g/kg
    freq: float
        frequency in GHz
    pci: np.array or xarray.DataArray
        correlation length in mm

    Returns
    -------
    gbih: np.array or xarray.DataArray
        2-flux scattering coefficient at h pol
    gbiv: np.array or xarray.DataArray
        2-flux scattering coefficient at v pol
    gs6: np.array or xarray.DataArray
        6-flux scattering coefficient
    ga2i: np.array or xarray.DataArray
        2-flux absorption coefficient

    Notes
    -----
    This function was introduced into the MEMLS code by `R.T. Tonboe <http://research.dmi.dk/staff/all-staff/rtt/>`_.
    It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    k=(2*3.14159)/(0.3 /freq)
    eice=3.15+0.002*1j
    if Ti.max() > 100.:
        Ti=Ti-273.15

    Sbr = pf.Sb(Ti) #inserted by Clara
    volb = pf.Vb(sal,Sbr)
    [eb,ebi] = ebrine(Ti,freq)
    ebri = eb-ebi*1j
    #%emis=eice_rn2p(eice,eb+ebi*i,volb);
    emis=eice_s2p(eice,eb+ebi*1j,volb)
    ags6=iborn_s2p(eice,ebri,emis,volb,k,pci*0.001)
    gs6=gs6-gs6*si+ags6*si
    return gbih,gbiv,gs6,ga2i



def scice_my(si,gbih,gbiv,gs6,ga2i,Ti,dens,freq,pci,sal):

    """Compute the scattering coefficient of multiyear ice from structural parameters

    This function computes the scattering coefficient of multiyear ice from structural parameters.

    Parameters
    ----------
    si: xarray.DataArray
        layer type ice/snow [1/0]
    gbih: xarray.DataArray
        2-flux scattering coefficient at h pol
    gbiv: xarray.DataArray
        2-flux scattering coefficient at v pol
    gs6: xarray.DataArray
        6-flux scattering coefficient
    ga2i: xarray.DataArray
        2-flux absorption coefficient
    Ti: xarray.DataArray
        temperature in K
    dens: xarray.DataArray
        density in g/cm3
    freq: float
        frequency in GHz
    pci: xarray.DataArray
        correlation length in mm
    sal: xarray.DataArray
        salinity in g/kg

    Returns
    -------
    gbih: xarray.DataArray
        2-flux scattering coefficient at h pol
    gbiv: xarray.DataArray
        2-flux scattering coefficient at v pol
    gs6: xarray.DataArray
        6-flux scattering coefficient
    ga2i: xarray.DataArray
        2-flux absorption coefficient

    Notes
    -----
    This function was introduced into the MEMLS code by `R.T. Tonboe <http://research.dmi.dk/staff/all-staff/rtt/>`_.
    It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    k=(2.*3.14159)/(0.3/freq)
    eice=3.15+0.002j
    #T=Ti-273.15;
    epsi=eice.real
    epsii=eice.imag

    #permittivity of saline ice
    [sepsi, sepsii] = sie(si,sal,Ti,freq,epsi,epsii)

    eice=sepsi+sepsii*1j

    vola=(0.926-dens)/0.926
    ### added by Clara , commented on 22.09., uncommented on 17.10.
    vola = vola.where(vola>0,0)
    ###
    emis=eice_s2p(eice,1.0+0.0j,vola)
    ags6=iborn_s2p(eice,1.0+0.0j,emis,vola,k,pci*0.001)
    gs6=gs6-gs6*si+ags6*si
    return gbih,gbiv,gs6,ga2i




def absorp2f(gbih,gbiv,gs6,ga2i,epsi,epsii,roi,Ti,pci,freq,Wi,gai):
    """Compute the absorption and scattering coefficient from structural parameters

    This function computes the absorption and scattering coefficient from structural parameters.

    Parameters
    ----------
    gbih: xarray.DataArray
        2-flux scattering coefficient at h pol
    gbiv: xarray.DataArray
        2-flux scattering coefficient at v pol
    gs6: xarray.DataArray
        6-flux scattering coefficient
    ga2i: xarray.DataArray
        2-flux absorption coefficient
    epsi: xarray.DataArray
        permittivity
    epsii: xarray.DataArray
        loss
    roi: xarray.DataArray
        density in g/cm3
    Ti: xarray.DataArray
        temperature in K
    pci: xarray.DataArray
        correlation length in mm
    freq: float
        frequency in GHz
    Wi: xarray.DataArray
        wetness between 0 and 1
    gai: xarray.DataArray
        absorption coefficient

    Returns
    -------
    gbih: xarray.DataArray
        2-flux scattering coefficient at h pol
    gbiv: xarray.DataArray
        2-flux scattering coefficient at v pol
    gs6: xarray.DataArray
        6-flux scattering coefficient
    ga2i: xarray.DataArray
        2-flux absorption coefficient

    Notes
    -----
    This function was introduced into the MEMLS code by `R.T. Tonboe <http://research.dmi.dk/staff/all-staff/rtt/>`_.
    It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    if roi.max() > 10:
      roi = roi/1000.0
    ## constants
    c = 2.99
    roair = 0.001293
    roice = 0.926
    ## specular component of scattering coefficient
    ## usually 0 can be important in new snow!
    dgb0h = 0
    dgb0v = 0
    ## aus der Theorie scattering coefficient
    k = freq*(2.*np.pi/0.299793)
    eice = 3.18
    vfi = roi/roice

    omega = np.sqrt((epsi - 1)/epsi)

    gb6 = 0.5* gs6 * (1-omega)
    gc6 = 0.25* gs6 * omega

    ## -> 2 Flux
    gtr = (4. * gc6) / (gai + 2.* gc6)
    ga2i = gai * (1 + gtr)

    gbih = (gb6 + dgb0h) + gtr * gc6
    gbiv = (gb6 + dgb0v) + gtr * gc6

    return gbih,gbiv,gs6,ga2i




def pfadc(teta,di,epsi,gs6):

    """Compute the effective path length in a layer

    This function computes the effective path length in a layer.

    Parameters
    ----------
    teta: float
        incidence angle at snow-air interface in degrees
    di: xarray.DataArray
        ice thickness in m
    epsi: xarray.DataArray
        permittivity
    gs6: xarray.DataArray
        6-flux scattering coefficient

    Returns
    -------
    dei: xarray.DataArray
        effective path length in m
    tei: xarray.DataArray
        local incidence angle
    tscat: xarray.DataArray
        tau scattering

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    N = len(epsi.layer_nb)
    ns = np.sqrt(epsi)
    costetasn = np.sqrt(1-(np.sin(teta)/ns)**2)
    cosc = np.sqrt(1-(1/ns)**2)
    costetasc = 0.5 * (1 + cosc)
    dei = di/costetasn

    #tauscat = zeros(len(epsi)+array([1,0]))
    layer_max = ns['layer_nb'].max()
    dim0 = ns.sel(layer_nb=layer_max)*0
    dim0['layer_nb'] = layer_max+1
    tauscat = xr.concat([epsi*0, dim0], dim='layer_nb')


    for m in range(N-1,-1,-1):
        tauscat.loc[dict(layer_nb=m+1)] = tauscat.isel(layer_nb=m+1) + dei.isel(layer_nb=m) * gs6.isel(layer_nb=m)/2
    tscat = np.exp(-1 * tauscat.isel(layer_nb=range(N)))
    costeta = tscat * costetasn + (1-tscat) * costetasc

    tei = np.arccos(costeta)
    tei*180/np.pi

    return dei,tei,tscat



def polmix(tscat,sih,siv):
    """Compute the polarization mixing of the interface reflectivities of each layer

    This function computes the polarization mixing of the interface reflectivities of each layer (taking into
    account the first order scattering)

    Parameters
    ----------
    tscat: xarray.DataArray
        tau scattering
    sih: xarray.DataArray
        interface reflectivity at h-pol
    siv: xarray.DataArray
        interface reflectivity at v-pol

    Returns
    -------
    sih: xarray.DataArray
        interface reflectivity at h-pol
    siv: xarray.DataArray
        interface reflectivity at v-pol

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input
    by `C. Burgard <http://www.github.com/ClimateClara>`_ to be used in ARC3O.
    """

    tscat = append_laydim_end(tscat,1)

    smean = 0.5 * (sih + siv)
    deltas = 0.5 * tscat.values * (sih.values - siv.values)
    sih = smean + deltas
    siv = smean - deltas
    return sih,siv



def rt(gai,gbi,dei):
    """Compute the layer reflectivity and transmissivity

    This function computes the layer reflectivity and transmissivity.

    Parameters
    ----------
    gai: xarray.DataArray
        absorption coefficient
    gbi: xarray.DataArray
        scattering coefficient
    dei: xarray.DataArray
        effective path length in m

    Returns
    -------
    ri: xarray.DataArray
        reflectivity
    ti: xarray.DataArray
        transmissivity

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    gamma = np.sqrt(gai * (gai + 2 * gbi))
    t0i = np.exp(gamma * dei * (-1))
    r0i = gbi.where(gbi>0.00001,0) / (gai.where(gbi>0.00001,0) + gbi.where(gbi>0.00001,0) + gamma.where(gbi>0.00001,1))
    t02 = t0i**2
    r02 = r0i**2
    ri = r0i * (1 - t02)/ (1 - t02 * r02)
    ti = t0i * (1 - r02) / (1 - t02 * r02)
    return ri, ti




def xr_diag(v,k=0):
    """Create diagonal matrix with values of v on the diagonal

    This function is more technical, creates diagonal matrix with values of v on the diagonal

    Parameters
    ----------
    v: xarray.DataArray
        xarray with the wished dimensions
    k: int
        coefficient if diagonal is shifted from middle

    Returns
    -------
    res: xarray.DataArray
        resulting diagonal matrix

    Notes
    -----
    This function was introduced by `C. Burgard <http://www.github.com/ClimateClara>`_ to adapt the python functions to
    :py:class:`xarray.DataArray` parameters and output.
    """

    ##########################
    ### new approach
    aa = v*0
    bb = xr.DataArray(np.zeros(len(v['layer_nb'])), coords=[('matrix_dim', v['layer_nb'])])
    res = aa*bb
    ##########################

#    aa = np.zeros((len(v.time),len(v.lat),len(v.lon),len(v.layer_nb),len(v.layer_nb)))
#    res = xr.DataArray(aa, coords=[
#                                    ('time', v.time.values),
#                                    ('lat',v.lat.values),
#                                    ('lon',v.lon.values),
#                                    ('layer_nb',v.layer_nb.values),
#                                    ('matrix_dim',v.layer_nb.values)])

    dim0 = v.dims
    if 'layer_nb' in dim0:
        n=len(v['layer_nb']) + k
        if k>=0:
            i=k
        else:
             i = (-k)*n

        for ll in range(1,n+1):
            #print(ll)
            res.loc[dict(layer_nb=ll,matrix_dim=ll+i)] = v.sel(layer_nb=ll)

    return res



def build_xarray(data,temp):
    """Transform :py:class:`np.array` into :py:class:`xarray.DataArray`

    This function is more technical, transforms :py:class:`np.array` into :py:class:`xarray.DataArray`.

    Parameters
    ----------
    data : np.array
        data to be transformed
    temp: xarray.DataArray
        other xarray.DataArray that has the wished output dimensions

    Returns
    -------
    res: xarray.DataArray
        resulting xarray.DataArray of same dimensions as `temp`

    Notes
    -----
    This function was introduced by `C. Burgard <http://www.github.com/ClimateClara>`_ to adapt the python functions to
    :py:class:`xarray.DataArray` parameters and output.
    """

    ####################################################
    #### new approach
    res = xr.DataArray(data, coords=temp.coords)
    ####################################################

#    #transform a np.array into xarray
#    res = xr.DataArray(data, coords=[
#                                ('time', temp.time.values),
#                                ('lat',temp.lat.values),
#                                ('lon',temp.lon.values),
#                                ('layer_nb',temp.layer_nb.values)])
    return res



def build_xarray_matrix2D(data,temp):
    """Transform a :py:class:`np.array` into :py:class:`xarray.DataArray` over the two dimensions ``layer_nb`` and
    ``matrix_dim`` for matrix operations

    This function is more technical, transforms a :py:class:`np.array` into :py:class:`xarray.DataArray` over the two dimensions
    ``layer_nb`` and ``matrix_dim`` for matrix operations.

    Parameters
    ----------
    data: np.array
        data to be transformed
    temp: xarray.DataArray
        other xarray.DataArray that has the wished dimensions

    Returns
    -------
    test: xarray.DataArray
        resulting xarray with two times dimension layer_nb for matrix operations

    Notes
    -----
    This function was introduced by `C. Burgard <http://www.github.com/ClimateClara>`_ to adapt the python functions to
    :py:class:`xarray.DataArray` parameters and output.
    """

    test = xr.DataArray(data, coords=[
                                ('layer_nb',temp.layer_nb.values),
                                ('matrix_dim',temp.layer_nb.values)])
    return test




def xr_eye(v,k=0):
    """Create diagonal matrix with ones on the diagonal

    This function is more technical, creates diagonal matrix with ones on the diagonal, gives out an :py:class:`xarray.DataArray`.

    Parameters
    ----------
    v: xarray.DataArray
        xarray.DataArray with the wished dimensions
    k: int
        coefficient if you want to shift the ones

    Returns
    -------
    test: xarray.DataArray
        resulting xarray with two times dimension layer_nb for matrix operations

    Notes
    -----
    This function was introduced by `C. Burgard <http://www.github.com/ClimateClara>`_ to adapt the python functions to
    :py:class:`xarray.DataArray` parameters and output.
    """

    ##########################
    ### new approach
    aa = v*0
    bb = xr.DataArray(np.zeros(len(v['layer_nb'])), coords=[('matrix_dim', v['layer_nb'])])
    res = aa*bb
    ##########################    



    dim0 = v.dims
    if 'layer_nb' in dim0:
        n=len(v['layer_nb']) + k
        if k>=0:
            i=k
        else:
             i = (-k)*n

        for ll in range(1,n+1):
            #print(ll)
            res.loc[dict(layer_nb=ll,matrix_dim=ll+i)] = 1.

    return res



def xr_matmul(A,B,input_dims,output_dims):
    """Compute matrix multiplication

    This function is more technical, matrix multiplication for :py:class:`xarray.DataArray`, :py:func:`xarray.dot` gave weird results

    Parameters
    ----------
    A: xarray.DataArray
        matrix 1
    B: xarray.DataArray
        matrix 2

    Returns
    -------
    asol: xarray.DataArray
        result of matrix multiplications

    Notes
    -----
    This function was introduced by `C. Burgard <http://www.github.com/ClimateClara>`_ to adapt the python functions to
    :py:class:`xarray.DataArray` parameters and output.
    """

    asol = xr.apply_ufunc(np.matmul,
                          A, B,
                          input_core_dims=input_dims,#[['layer_nb','matrix_dim'],['layer_nb','matrix_dim']],
                          output_core_dims=output_dims#[['layer_nb', 'matrix_dim']]
                         )

    return asol

def xr_linalg_inv(A):

    """Compute the (multiplicative) inverse of a matrix.

    This function is more technical, equivalent of :py:func:`numpy.linalg.inv` for :py:class:`xarray.DataArray`

    Parameters
    ----------
    A: xarray.DataArray
        matrix to be inverted

    Returns
    -------
    res: xarray.DataArray
        inverted matrix

    Notes
    -----
    This function was introduced by `C. Burgard <http://www.github.com/ClimateClara>`_ to adapt the python functions to
    :py:class:`xarray.DataArray` parameters and output.
    """

    res = xr.apply_ufunc(np.linalg.inv,
                         A,
                         input_core_dims=[['layer_nb','matrix_dim']],
                         output_core_dims=[['layer_nb', 'matrix_dim']])

    return res





def add_matrix_dim(A,name_new_dim):
    """Add a dummy ``matrix_dim`` to enable matrix multiplication

    This function is more technical, adds a dummy ``matrix_dim`` to enable matrix multiplication

    Parameters
    ----------
    A: xarray.DataArray
        matrix to be changed

    Returns
    -------
    res: xarray.DataArray
        changed matrix

    Notes
    -----
    This function was introduced by `C. Burgard <http://www.github.com/ClimateClara>`_ to adapt the python functions to
    :py:class:`xarray.DataArray` parameters and output.
    """

    res = A*xr.DataArray(np.ones(1),coords=[(name_new_dim, range(1,2))])

    return res


def layer(ri,s_i,ti,Ti,Tgnd,Tsky):
    """Compute the upwelling brightness temperatures

    This function computes the upwelling brightness temperatures, see Eq. 14 in :cite:`wiesmann98`.

    Parameters
    ----------
    ri: xarray.DataArray
        reflectivity
    s_i: xarray.DataArray
        interface reflectivity
    ti: xarray.DataArray
        transmissivity
    Ti: xarray.DataArray
        temperature in K
    Tgnd: float
        brightness temperature of the ocean below the ice in K
    Tsky: float
        brightness temperature of the sky in K

    Returns
    -------
    D1: xarray.DataArray
        brightness temperatures at each layer

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was translated to Python and adapted for multi-dimensional input by `C. Burgard <http://www.github.com/ClimateClara>`_
    to be used in ARC3O.
    """

    N = len(ri.layer_nb)
    ei = 1 - ri - ti

    E = ei * Ti
    E.loc[dict(layer_nb=1)] = E.isel(layer_nb=0) + ri.isel(layer_nb=0)* (1 - s_i.isel(layer_nb=0)) * Tgnd
    E.loc[dict(layer_nb=N)] = E.isel(layer_nb=N-1) + ti.isel(layer_nb=N-1)* (1 - s_i.isel(layer_nb=N))* Tsky

    F = ei * Ti
    F.loc[dict(layer_nb=1)] = F.isel(layer_nb=0) + ti.isel(layer_nb=0)* (1 - s_i.isel(layer_nb=0)) * Tgnd
    F.loc[dict(layer_nb=N)] = F.isel(layer_nb=N-1) + ri.isel(layer_nb=N-1)* (1 - s_i.isel(layer_nb=N))* Tsky

    aa1 = build_xarray(ri.values*s_i.isel(layer_nb=range(N)).values,ri)
    M1 = xr_diag(aa1) #could not change it with apply_ufunc
    aah = build_xarray(ti.isel(layer_nb=range(N-1)).values*(1-s_i.isel(layer_nb=range(1,N)).values),ti.isel(layer_nb=range(N-1))) #could not change it easily
    H = xr_diag(aah) #could not change it with apply_ufunc
    M1.loc[dict(layer_nb=range(1,N),matrix_dim=range(2,N+1))] = M1.isel(layer_nb=range(N-1),matrix_dim=range(1,N)).values + H.values

    aa2 = build_xarray(ti.values*s_i.isel(layer_nb=range(1,N+1)).values,ti)
    M2 = xr_diag(aa2)
    aah = build_xarray(ri.isel(layer_nb=range(1,N)).values*(1-s_i.isel(layer_nb=range(1,N)).values),ri.isel(layer_nb=range(N-1)))
    H = xr_diag(aah)
    M2.loc[dict(layer_nb=range(2,N+1),matrix_dim=range(1,N))] = M2.isel(layer_nb=range(1,N),matrix_dim=range(N-1)).values + H.values

    aa3 = build_xarray(ti.values*s_i.isel(layer_nb=range(N)).values,ti)
    M3 = xr_diag(aa3)
    aah = build_xarray(ri.isel(layer_nb=range(N-1)).values*(1-s_i.isel(layer_nb=range(1,N)).values),ri.isel(layer_nb=range(N-1)))
    H = xr_diag(aah)
    M3.loc[dict(layer_nb=range(1,N),matrix_dim=range(2,N+1))] = M3.isel(layer_nb=range(N-1),matrix_dim=range(1,N)).values + H.values

    aa4 = build_xarray(ri.values*s_i.isel(layer_nb=range(1,N+1)).values,ri)
    M4 = xr_diag(aa4)
    aah = build_xarray(ti.isel(layer_nb=range(1,N)).values*(1-s_i.isel(layer_nb=range(1,N)).values),ti.isel(layer_nb=range(N-1)))
    H = xr_diag(aah)
    M4.loc[dict(layer_nb=range(2,N+1),matrix_dim=range(1,N))] = M4.isel(layer_nb=range(1,N),matrix_dim=range(N-1)).values + H.values

    I = xr_eye(ri)

    M5 = xr_matmul(M3,xr_matmul(xr_linalg_inv(I - M1),M2,[['layer_nb','matrix_dim'],['layer_nb','matrix_dim']],[['layer_nb','matrix_dim']]),[['layer_nb','matrix_dim'],['layer_nb','matrix_dim']],[['layer_nb','matrix_dim']]) + M4
    input_dims = [['layer_nb','matrix_dim'],['layer_nb','matrix_dim2']]
    output_dims = [['layer_nb','matrix_dim2']]
    D = xr_matmul(xr_linalg_inv(I - M5),xr_matmul(M3,xr_matmul(xr_linalg_inv(I - M1),add_matrix_dim(E,'matrix_dim2'),input_dims,output_dims),input_dims,output_dims) + F, input_dims,output_dims)
    D1 = D.isel(matrix_dim2=0).drop('matrix_dim2')
    return D1




def memls_2D_1freq(freq,DI,TI,WI,ROI,PCI,SAL,SITYPE):

    """Compute the brightness temperature of a lon-lat-time-depth field at a given frequency

    This is the main MEMLS function. It computes the brightness temperature of a lon-lat-time-depth field at a given frequency.

    Parameters
    ----------
    freq: float
        frequency
    DI: xarray.DataArray
        ice thickness in m
    TI: xarray.DataArray
        temperature in K
    WI: xarray.DataArray
        wetness between 0 and 1
    ROI: xarray.DataArray
        density in kg/m3
    PCI: xarray.DataArray
        correlation length in mm
    SAL: xarray.DataArray
        salinity in g/kg
    SITYPE: xarray.DataArray
        snow 1 /first year ice 3 /multiyear ice 4

    Returns
    -------
    Tbh: xarray.DataArray
        brightness temperature h-pol
    Tbv: xarray.DataArray
        brightness temperature v-pol
    eh: xarray.DataArray
        emissivity h-pol
    ev:	xarray.DataArray
        emissivity v-pol

    Notes
    -----
    This function is part of the original MEMLS developed by the Institute of Applied Physics,
    University of Bern, Switzerland. A description of that model version can be found in :cite:`wiesmann98`
    and :cite:`wiesmann99`. It was slightly modified to accomodate sea-ice layers by
    `R.T. Tonboe <http://research.dmi.dk/staff/all-staff/rtt/>`_. It was translated to Python, adapted for multi-dimensional input and
    further extended by `C. Burgard <http://www.github.com/ClimateClara>`_ to be used in ARC3O.
    """

    start_time = timeit.default_timer()
    print('--------------COMPUTING BRIGHTNESS TEMPERATURE FOR '+str(freq)+'GHz--------------')

    #### CHANGE DEPENDING ON WHAT INPUT WE WANT
    teta=55
    s0h=0.75
    s0v=0.25
    Tsky=0
    Tgnd=271.35
    sccho=2 #4: iborn for spheres, 5: iborn for shells and spheres, now a function of type.
    #################

    di = DI.copy()#/100.
    Ti = TI.copy()
    Wi = WI.copy()
    roi = ROI.copy()/1000.
    pci = PCI.copy()
    sal = SAL.copy()
    si = SITYPE.copy()

    #si0 = si*0
    #si0 = si0.where(si==1,1)
    teta = (teta * np.pi) / 180

    layer_max = di['layer_nb'].max()
    ############################
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    start_time = timeit.default_timer()
    print('--------------ABSORPTION '+str(freq)+'GHz--------------')
    [epsi,epsii] = ro2epsd(roi,Ti,freq)
    [epsi,epsii] = mixmod(freq,Ti,Wi,epsi,epsii)
    [epsi,epsii] = sie(si.where(si==3,2)-2,sal,Ti,freq,epsi,epsii)
    [epsi,epsii] = mysie(si.where(si==4,3)-3,roi,Ti,sal,freq,epsi,epsii)
    gai = abscoeff(epsi,epsii,Ti,freq) #ga is absorption coefficient
    ns = np.sqrt(epsi) #real part of the refractive index of the slab
    tei = tei_ndim(teta,ns) #np.arcsin(...) is the critical angle for total reflection
    dei = pfadi(tei,di)
    #add a 1 to epsi to have same dimensions for tei and epsi, get fresnel
    #coefficients (modified to fresnelc0 because conflict with other functions)
    [sih,siv] = fresnelc0(tei,xr.concat([epsi, epsi.sel(layer_nb=layer_max)*0+1], dim='layer_nb'))

    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    start_time = timeit.default_timer()
    print('--------------SCATTERING '+str(freq)+'GHz--------------')
    [gbih,gbiv,gs6,ga2i] = sccoeff(roi,Ti,pci,freq,Wi,gai,sccho)
    #for snow
    [gbih,gbiv,gs6,ga2i] = meteo_sc(si.where(si==1,0)*1,roi,Ti,pci,freq,Wi,gai,gbih,gbiv,gs6,ga2i)
    #for fyi
    [gbih,gbiv,gs6,ga2i] = scice(si.where(si==3,2)-2,gbih,gbiv,gs6,ga2i,Ti,sal,freq,pci)
    #for myi
    [gbih,gbiv,gs6,ga2i] = scice_my(si.where(si==4,3)-3,gbih,gbiv,gs6,ga2i,Ti,roi,freq,pci,sal)
    [gbih,gbiv,gs6,ga2i] = absorp2f(gbih,gbiv,gs6,ga2i,epsi,epsii,roi,Ti,pci,freq,Wi,gai)
    [dei,tei,tscat] = pfadc(teta,di,epsi,gs6)

    sih = append_laydim_begin(sih,s0h)
    siv = append_laydim_begin(siv,s0v)
    [sih,siv] = polmix(tscat,sih,siv)

    #### was in my memls, let's see what's the result if I leave it out, maybe this: DataArray.interpolate_na could be of interest
    # if len(ri) > 1:
    #     for i in range(len(ri)-2,-1,-1):
    #       if np.isnan(ri[i])==True:
    #         ri[i] = ri[i+1]
    #       if np.isnan(ti[i])==True:
    #         ti[i] = ti[i+1]

    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    start_time = timeit.default_timer()
    print('--------------WRAP-UP '+str(freq)+'GHz--------------')
    #layer reflectivity and transmissivity (horizontal)
    [ri,ti]  = rt(ga2i,gbih,dei)
    #Brightness temperature at every layer (horizontal)
    Dh   = layer(ri,sih,ti,Ti,Tgnd,Tsky)
    # amount of layers
    N = len(Ti.layer_nb)
    #TBH at top
    Tbh = (1-sih.isel(layer_nb=N))*Dh.isel(layer_nb=N-1) + sih.isel(layer_nb=N)*Tsky
    print('H-pol done')


    #layer reflectivity and transmissivity (vertical)
    [ri,ti]  = rt(ga2i,gbiv,dei)
    #Brightness temperature at every layer (vertical)
    Dv   = layer(ri,siv,ti,Ti,Tgnd,Tsky)
    # amount of layers
    N = len(Ti.layer_nb)
    Tbv = (1-siv.isel(layer_nb=N))*Dv.isel(layer_nb=N-1) + siv.isel(layer_nb=N)*Tsky
    print('V-pol done')

    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    start_time = timeit.default_timer()
    print('--------------COMPUTING EMISSIVITIES '+str(freq)+'GHz --------------')
    if Tsky==0:
        eDh = Dh.copy()
    else:
        eDh = layer(ri,sih,ti,Ti,Tgnd,0)
    eTbh = (1-sih.isel(layer_nb=N))*eDh.isel(layer_nb=N-1)
    yh0 = eTbh

    eDh = layer(ri,sih,ti,Ti,Tgnd,100)
    eTbh = (1-sih.isel(layer_nb=N))*eDh.isel(layer_nb=N-1) + sih.isel(layer_nb=N)*100.
    yh100 = eTbh

    print('H-pol done')

    if Tsky==0:
        eDv = Dv.copy()
    else:
        eDv = layer(ri,siv,ti,Ti,Tgnd,0)
    eTbv = (1-siv.isel(layer_nb=N))*eDv.isel(layer_nb=N-1)
    yv0 = eTbv

    eDv   = layer(ri,siv,ti,Ti,Tgnd,100)
    eTbv = (1-siv.isel(layer_nb=N))*eDv.isel(layer_nb=N-1) + siv.isel(layer_nb=N)*100.
    yv100 = eTbv

    print('V-pol done')


    #Calculate emissivities, formulas can be found in Wiesmann and Mätzler, 1999
    rv = (yv100 - yv0)/100.
    rh = (yh100 - yh0)/100.
    ev = 1 - rv
    eh = 1 - rh

    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    print('--------------MEMLS '+str(freq)+'GHz FINISHED--------------')

    #return Tbh.drop('layer_nb').drop('matrix_dim'),Tbv.drop('layer_nb').drop('matrix_dim'),eh.drop('layer_nb').drop('matrix_dim'),ev.drop('layer_nb').drop('matrix_dim')
    return Tbh.drop('layer_nb'),Tbv.drop('layer_nb'),eh.drop('layer_nb'),ev.drop('layer_nb')
