
import GridWorldHelpers as gwh




def init_Tstandard(d, colors = [0,1,2], init_value = 0): 
    """
    
    Parameters
    ----------
    d : dimension of grid
    
    colors : list of color codes
    
    init_value: count to initialize with
    
    Returns
    -------
    Tstandard: a dict where keys are (S_t,A_t) aka tuples of the form ((y,x,c),(a1,a2))
    and values are dictionaries with keys formed by (S_{t+1}) aka tuples of the form
    (y,x,c)
    
    Example:
        
        Tstandard[((1,1,1),(0,1))][(0,0,0)]
        
        accesses the count for times was at (1,1) on the grid with color 1, 
        took action "up" (0,1) and ended up at (0,0,0)
        
        *Note that in practice many transitions are impossible and should stay at 0
        but we initialize with all possibilities so that this matrix is (a) not world
        specific and (b) to reflect that we do not actually give the agent this logical
        information
    
    
    #TODO: think about consequences of how initialize this -- if do all to 0, then good
    because those that are truly 0 will stay 0 but bad because will never impute a transition
    until have observed it at least once. Maybe you want to initilize this to \epsilon?
    
    
    """
    # get possible actions
    action_descs = gwh.load_actions()
    
    inner_S_dict = {((i,j,c)) : init_value for i in range(d) for j in range(d) for c in colors}


    T = {((i, j, c), action) : inner_S_dict for i in range(d) for j in range(d) 
                for c in colors for action in list(action_descs.keys())}

    return(T)    
    
    
def init_Tmice(d, colors = [0,1,2], init_value = 0 ):
    """

    Parameters
    ----------
    d : dimension of grid
    
    colors : list of color codes
    
    init_value: count to initialize with
    
    Returns
    -------
    dict with entries (y,x,c) where for "x" value is:
           
        dict where keys are tuples of the form ((y,x,c),(a1,a2),(y,c))
        and values are dictionaries with keys 0,...,d-1 for each of the possible
        values of x
        
    for "y" value, it is same only keys are tuples of form ((y,x,c),(a1,a2),(x,c))
    
    for "c" value, it is same only keys are tuples of form ((y,x,c),(a1,a2),(y,x))


    """
    action_descs = gwh.load_actions()
  
    inner_xy_dict = {i: init_value for i in range(d)}
    inner_c_dict = {c: init_value for c in colors}

    x_dict = {((i, j, c), action, (u,v)) : inner_xy_dict for i in range(d) for j in range(d) 
                for c in colors for action in list(action_descs.keys()) for u in range(d) for v in colors}

    y_dict = x_dict.copy() #same structure 
    
    c_dict = {((i, j, c), action, (u,v)) : inner_c_dict for i in range(d) for j in range(d) 
                for c in colors for action in list(action_descs.keys()) for u in range(d) for v in range(d)}


    return {"x":x_dict,"y":y_dict, "c":c_dict}
    
    
    
    
    