import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def fsag ( r , x1 , x2 ):
    return np.sqrt(r ** 2 - x1 ** 2) - np.sqrt(r ** 2 - x2 ** 2)


def invsag ( ss , y1 , y2 ):
    m1 = ss ** 2 + (y1 - y2) ** 2
    m2 = ss ** 2 + (y1 + y2) ** 2
    return np.sqrt(m1 * m2)/ (2 * ss)


def fr ( index , bc , power , ct ):
    m1 = (1000 * (index - 1) * bc) / (power * bc + 1000 * (index - 1))
    return (ct * (index - 1) / index) + m1


def sag_asph ( r , k , y1 , y2 ):
    m1 = np.sqrt(r ** 2 - k * y1 ** 2 - y1 ** 2)
    m2 = np.sqrt(r ** 2 - k * y2 ** 2 - y2 ** 2)
    return (m1 - m2) / (k + 1)


def asph_parms ( Dia , boz , ael , bc ):
    sag1 = fsag(bc , 0 , boz / 2)
    sag2 = fsag(bc , 0 , Dia / 2) - ael
    y1 , y2 = boz / 2 , Dia / 2
    r0_asph = -(sag1 ** 2 * y2 ** 2 - sag2 ** 2 * y1 ** 2) / (2 * sag1 * sag2 ** 2 - 2 * sag1 ** 2 * sag2)
    k_asph = -(-sag1 ** 2 * sag2 + sag1 * sag2 ** 2 + sag1 * y2 ** 2 - sag2 * y1 ** 2) / (
            sag1 * sag2 ** 2 - sag1 ** 2 * sag2)
    return r0_asph , k_asph

def calcs ():
    st.header('Bifocal Design')
    st.sidebar.header('Contact Lens Design App')
    st.sidebar.header('Input Parameters')

    def user_input ():
        refindex = st.sidebar.text_input(label="Refractive Index" , value=1.43)
        Dia = st.sidebar.text_input(label="Diameter" , value=10.)
        BC = st.sidebar.text_input(label="Base Curve" , value=8.)
        AEL = st.sidebar.text_input(label="Axial Edge Lift" , value=0.12)
        BOZ = st.sidebar.text_input(label="Back Optic Zone" , value=8.2)
        ct = st.sidebar.text_input(label="Center Thickness" , value=0.2)
        et = st.sidebar.text_input(label="Edge Thickness" , value=0.22)
        FOZ1 = st.sidebar.text_input(label="Front Optic Zone 1 (Central)" , value=3.0)
        p1 = st.sidebar.text_input(label="Power 1 (Central)" , value=3.0)
        FOZ2 = st.sidebar.text_input(label="Front Optic Zone 2 (Peripheral)" , value=6.0)
        p2 = st.sidebar.text_input(label="Power 2 (Peripheral)" , value=1.0)
        data = {'RI': refindex , 'Dia': Dia , 'BC': BC , 'AEL': AEL , 'BOZ': BOZ , 'CT': ct , 'ET': et , 'FOZ 1': FOZ1 ,
                'P1': p1 , 'FOZ 2': FOZ2 , 'P2': p2}
        features = pd.DataFrame(data , index=[0])
        return features

    df = user_input()
    st.subheader('User inputs')
    st.write(df)
    df = df.to_numpy()[0]
    st.sidebar.write('Created by paymanrajai@gmail.com')

    # vars
    RI = float(df[0])
    Dia = float(df[1])
    bc = float(df[2])
    ael = float(df[3])
    boz = float(df[4])
    ct = float(df[5])
    et = float(df[6])
    foz1 = float(df[7])
    p1 = float(df[8])
    foz2 = float(df[9])
    p2 = float(df[10])
    jt_inter = 0.15

    asphparms = asph_parms(Dia , boz , ael , bc)

    r0_asph = asphparms[0]
    k_asph = asphparms[1]

    def backsag(x):
        out = np.array([])
        ozsag = fsag(bc , 0 , boz / 2)
        if np.abs(x) <= boz/2 :
            out1 = np.append(out , fsag(bc , 0 , np.abs(x)) + ct)
        else:
            out1 = np.append(out , ozsag + sag_asph(r0_asph , k_asph , boz / 2 , np.abs(x)) + ct)
        return out1

    fr1 = fr(RI , bc , p1 , ct)


    def solve_eqs ( vars ):
        ct2 , fr2 = vars
        eq1 = fr(RI , bc , p2 , ct2) - fr2
        eq2 = fsag(bc , 0 , foz2 / 2) + ct - fsag(fr2 , 0 , foz2 / 2) - ct2 - jt_inter
        return [eq1 , eq2]

    ct2 , fr2 = np.round(fsolve(solve_eqs , (0.2 , fr1)) , 3)
    delsag0 = fsag(fr1 , 0 , foz1 / 2) + fsag(fr2 , foz1 / 2 , foz2 / 2)
    delsag = backsag(Dia / 2) - et - delsag0
    fr3 = invsag(delsag , foz2 / 2 , Dia / 2).item()

    def frontsag ( x ):
        m1 = fsag(fr1 , 0 , foz1 / 2)
        out = np.array([])
        if np.abs(x) <= foz1 / 2:
            out1= np.append(out, fsag(fr1 , 0 , np.abs(x)))
        elif (np.abs(x) > foz1 / 2) & (np.abs(x) <= foz2 / 2):
            out1= np.append(out, fsag(fr2 , foz1 / 2 , np.abs(x)) + m1)
        else:
            out1= np.append(out, fsag(fr3 , foz2 / 2 , np.abs(x)) + delsag0)
        return out1

    def thickness(x):
        return backsag(x)- frontsag(x)
    jt1 = backsag(foz1 / 2) - frontsag(foz1 / 2)
    jt2 = backsag(foz2 / 2) - frontsag(foz2 / 2)



    x = np.linspace(0 , Dia / 2 , 200)

    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=True )

    # ax.set_aspect('equal')
    # # ax.axis('equal')
    ax1.plot(x , -np.array(list(map(frontsag, x))) , color='r' , label='Anterior')
    ax1.plot(x , -np.array(list(map(backsag, x))) , color='b' , label='Posterior')
    ax1.axvline(x= foz1/2, linestyle = '--', linewidth=0.8, color='g')
    ax1.axvline(x= foz2/2, linestyle = '--', linewidth=0.8, color='g' )
    ax1.set_xlim([0 , Dia / 2])
    ax1.set_ylabel('Sag (mm)')
    ax1.legend(loc = 'lower left')


    ax2.plot(x, np.array(list(map(thickness, x))), color='black')
    ax2.set_xlim([0 , Dia / 2])
    ax2.set_ylabel('Thickness (mm)')
    ax2.set_xlabel('Semi Chord (mm)')
    ax2.axvline(x= foz1/2, linestyle = '--', linewidth=0.8, color='g', label = 'FOZ 1')
    ax2.axvline(x= foz2/2, linestyle = '--', linewidth=0.8, color='g', label = 'FOZ 2' )
    ax2.legend(loc = 'upper left')


    st.pyplot(fig)
    output = {'Radius 1': np.round(fr1, 2),
              'Radius 2': np.round(fr2, 2),
              'Radius 3': np.round(fr3 , 2),
              'jt1': np.round(jt1, 2), 'jt2': np.round(jt2, 2)}
    output = pd.DataFrame(output , index=[0])
    st.table(output)
    st.write('jt1 = junction thickness at FOZ 1')
    st.write('jt2 = junction thickness at FOZ 2')


if __name__ == '__main__':
    calcs()
