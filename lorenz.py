import lorenz_cpp
import numpy as np
from scipy.interpolate import CubicHermiteSpline

class DotDict(dict):
    """A dictionary that allows dot notation access to its keys."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

def simulaLorenzCPP(r0, tTransiente, tSpan,
                    varNum, dir, crossVal,
                    sigma=10., beta=8./3., rho=28.,
                    maxCross=100, rtol=1e-7, atol=1e-10,
                    dense_output=True, trajectory=False):
    '''
    Integra o sistema de Lorenz com o integrador em C++ com a detecção de Eventos.

    **Tipos de Evento:**
    * **Corte do Plano em x:** varNum = "x" or 1
    * **Corte do Plano em y:** varNum = "y" or 2
    * **Corte do Plano em z:** varNum = "z" or 3
    * **Corte do Plano em x':** varNum = "x'" or "x`" or -1
    * **Corte do Plano em y':** varNum = "y'" or "y`" or -2
    * **Corte do Plano em z':** varNum = "z'" or "z`" or -3

    **Parâmetros:**
    * `r0` (array_like): Estado inicial r(t0).
    * `tTransiente` (float): Duração do Transiente a ser ignorado. Se menor ou igual a 0 não há transiente
    * `tSpan (float)`: Duração máxima da Integracão.
    * `varNum` (int or str): Define o tipo de evento a ser verificado. 0 para não buscar nenhum evento.
    * `dir` (int): Direção da verificação dos eventos (1 ou -1).
    * `crossVal` (float): Valor do plano sendo verificado pelos eventos.
    * `maxCross` (int): Número máximo de eventos a serem identificados, interrompendo a integração após serem identificados. 
                        maxCross <= 0 indica a busca de todos os eventos, com a integração sendo interrompida no tSpan.
    * `sigma`, `beta`, `rho` (float): Parâmetros do sistema de Lorenz.
    * `rtol`, `atol` (float): Tolerâncias relativa e absoluta para o DOP853.
    * `dense_output` (bool): Indica se o interpolador da integração deve ser retornado.
    * `trajectory` (bool): Indica se os passos de integração devem ser retornados.

    **Retorno:**
    * `DotDict`: Dicionário com atributos contando as arrays de estados (se trajectory==True), eventos (se varNum != 0) interpolador (se dense_output==True) e status da integração.

    '''
    
    # Verificação do evento a ser identificado
    if isinstance(varNum, str):
        fator = -1 if ("'" in varNum) or ("`" in varNum) else 1
        if 'x' in varNum.lower():
            varNum = fator * 1
        elif 'y' in varNum.lower():
            varNum = fator * 2
        elif 'z' in varNum.lower():
            varNum = fator * 3
        else:
            raise Exception("String varNum de Evento Inválida.")
    elif isinstance(varNum, int):
        if abs(varNum) > 3:
            raise Exception("Int varNum de Evento Inválido.")
    else:
        raise Exception("Evento varNum Inválido")

    if tTransiente <= 0:
        if isinstance(tSpan, (list, tuple)):
            bp = [tSpan[0], tSpan[-1]]
        else:
            bp = [0, tSpan]
    else:
        if isinstance(tSpan, (list, tuple)):
            bp = [-tTransiente, 0, tSpan[-1]]
        else:
            bp = [-tTransiente, 0, tSpan]
    
    # t, x, y, z, dx, dy, dz, tEv, xEv, yEv, zEv, status
    result = lorenz_cpp.run_simulation(r0=r0, breakpoints=bp,sigma=sigma, beta=beta, rho=rho, 
                                       varNum=varNum, dir=dir, crossVal=crossVal,max_cross=maxCross, 
                                       trackerStartTime=bp[-2], rtol=rtol, atol=atol)
    # Passos de Integração
    t = result[0]
    r = np.vstack(result[1:4]) # np.vstack((x,y,z)) # x, y, z = result[1:4]

    # Eventos Identificados
    tEv = result[7] # Tempos dos Eventos identificados
    rEv = np.vstack(result[8:11]) # np.vstack((xEv,yEv,zEv)) # xEv, yEv, zEv = result[8:11]

    sol = DotDict()
    if varNum != 0: # Salva os eventos
        sol['t_events'] = tEv
        sol['y_events'] = rEv
    if trajectory: # Salva os passos de Integração
        sol['t'] = t
        sol['y'] = r
    if dense_output: # Salva o interpolador da Integração
        dr = np.vstack(result[4:7]) # np.vstack((dx,dy,dz)) # dx, dy, dz = result[4:7]
        spline = CubicHermiteSpline(t, r, dr)
        sol['sol'] = spline
    sol['status'] = result[-1]
    return sol