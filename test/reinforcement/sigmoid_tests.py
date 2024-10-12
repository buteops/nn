import numpy as np

        
def basic_sigmoid_test(target):
    x = 1
    expected_output = 0.7310585786300049
    test_cases = [
        {
            "name": "datatype_check",
            "input": [x],
            "expected": float,
            "error": "Datatype mismatch."
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    test(test_cases, target)
         
def sigmoid_test(target):
    x = np.array([1, 2, 3])
    expected_output = np.array([0.73105858,
                                0.88079708,
                                0.95257413])
    test_cases = [
        {
            "name":"datatype_check",
            "input": [x],
            "expected": np.ndarray,
            "error":"Datatype mismatch."
        },
        {
            "name": "shape_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    test(test_cases, target)
    
            
        
def sigmoid_derivative_test(target):
    x = np.array([1, 2, 3])
    expected_output = np.array([0.19661193,
                                0.10499359,
                                0.04517666])
    test_cases = [
        {
            "name":"datatype_check",
            "input": [x],
            "expected": np.ndarray,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    test(test_cases, target)

def image2vector_test(target):
    image = np.array([[[ 0.67826139,  0.29380381],
                      [ 0.90714982,  0.52835647],
                      [ 0.4215251 ,  0.45017551]],

                     [[ 0.92814219,  0.96677647],
                      [ 0.85304703,  0.52351845],
                      [ 0.19981397,  0.27417313]],

                     [[ 0.60659855,  0.00533165],
                      [ 0.10820313,  0.49978937],
                      [ 0.34144279,  0.94630077]]])
    
    expected_output = np.array([[ 0.67826139],
                                [ 0.29380381],
                                [ 0.90714982],
                                [ 0.52835647],
                                [ 0.4215251 ],
                                [ 0.45017551],
                                [ 0.92814219],
                                [ 0.96677647],
                                [ 0.85304703],
                                [ 0.52351845],
                                [ 0.19981397],
                                [ 0.27417313],
                                [ 0.60659855],
                                [ 0.00533165],
                                [ 0.10820313],
                                [ 0.49978937],
                                [ 0.34144279],
                                [ 0.94630077]])
    test_cases = [
        {
            "name":"datatype_check",
            "input": [image],
            "expected": np.ndarray,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [image],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [image],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)

def normalizeRows_test(target):
    x = np.array([[0, 3, 4],
                  [1, 6, 4]])
    expected_output = np.array([[ 0., 0.6, 0.8 ],
                                [ 0.13736056, 0.82416338, 0.54944226]])
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [x],
            "expected": np.ndarray,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)       
        
def softmax_test(target):
    x = np.array([[9, 2, 5, 0, 0],
                  [7, 5, 0, 0 ,0]])
    expected_output = np.array([[ 9.80897665e-01, 8.94462891e-04,
                                 1.79657674e-02, 1.21052389e-04,
                                 1.21052389e-04],
                                
                                [ 8.78679856e-01, 1.18916387e-01,
                                 8.01252314e-04, 8.01252314e-04,
                                 8.01252314e-04]])
    test_cases = [
        {
            "name":"datatype_check",
            "input": [x],
            "expected": np.ndarray,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)

def L1_test(target):
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    expected_output = 1.1
    test_cases = [
        {
            "name":"datatype_check",
            "input": [yhat, y],
            "expected": float,
            "error":"The function should return a float."
        },
        {
            "name": "equation_output_check",
            "input": [yhat, y],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)
    
def L2_test(target):
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    expected_output = 0.43
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [yhat, y],
            "expected": float,
            "error":"The function should return a float."
        },
        {
            "name": "equation_output_check",
            "input": [yhat, y],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)
    
def test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            if test_case['name'] == "datatype_check":
                assert isinstance(target(*test_case['input']),
                                  test_case["expected"])
                success += 1
            if test_case['name'] == "equation_output_check":
                assert np.allclose(test_case["expected"],
                                   target(*test_case['input']))
                success += 1
            if test_case['name'] == "shape_check":
                assert test_case['expected'].shape == target(*test_case['input']).shape
                success += 1
        except:
            print("Error: " + test_case['error'])
            
    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError("Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(target.__name__))

def sigmoid_test(target):
    x = np.array([0, 2])
    output = target(x)
    assert type(output) == np.ndarray, "Wrong type. Expected np.ndarray"
    assert np.allclose(output, [0.5, 0.88079708]), f"Wrong value. {output} != [0.5, 0.88079708]"
    output = target(1)
    assert np.allclose(output, 0.7310585), f"Wrong value. {output} != 0.7310585"
    print('\033[92mAll tests passed!')
    
            
        
def initialize_with_zeros_test_1(target):
    dim = 3
    w, b = target(dim)
    assert type(b) == float, f"Wrong type for b. {type(b)} != float"
    assert b == 0., "b must be 0.0"
    assert type(w) == np.ndarray, f"Wrong type for w. {type(w)} != np.ndarray"
    assert w.shape == (dim, 1), f"Wrong shape for w. {w.shape} != {(dim, 1)}"
    assert np.allclose(w, [[0.], [0.], [0.]]), f"Wrong values for w. {w} != {[[0.], [0.], [0.]]}"
    print('\033[92mFirst test passed!')
    
def initialize_with_zeros_test_2(target):
    dim = 4
    w, b = target(dim)
    assert type(b) == float, f"Wrong type for b. {type(b)} != float"
    assert b == 0., "b must be 0.0"
    assert type(w) == np.ndarray, f"Wrong type for w. {type(w)} != np.ndarray"
    assert w.shape == (dim, 1), f"Wrong shape for w. {w.shape} != {(dim, 1)}"
    assert np.allclose(w, [[0.], [0.], [0.], [0.]]), f"Wrong values for w. {w} != {[[0.], [0.], [0.], [0.]]}"
    print('\033[92mSecond test passed!')    

def propagate_test(target):
    w, b = np.array([[1.], [2.], [-1]]), 2.5, 
    X = np.array([[1., 2., -1., 0], [3., 4., -3.2, 1], [3., 4., -3.2, -3.5]])
    Y = np.array([[1, 1, 0, 0]])

    expected_dw = np.array([[-0.03909333], [ 0.12501464], [-0.99960809]])
    expected_db = np.float64(0.288106326429569)
    expected_grads = {'dw': expected_dw,
                      'db': expected_db}
    expected_cost = np.array(2.0424567983978403)
    expected_output = (expected_grads, expected_cost)
    
    grads, cost = target( w, b, X, Y)

    assert type(grads['dw']) == np.ndarray, f"Wrong type for grads['dw']. {type(grads['dw'])} != np.ndarray"
    assert grads['dw'].shape == w.shape, f"Wrong shape for grads['dw']. {grads['dw'].shape} != {w.shape}"
    assert np.allclose(grads['dw'], expected_dw), f"Wrong values for grads['dw']. {grads['dw']} != {expected_dw}"
    assert np.allclose(grads['db'], expected_db), f"Wrong values for grads['db']. {grads['db']} != {expected_db}"
    assert np.allclose(cost, expected_cost), f"Wrong values for cost. {cost} != {expected_cost}"
    print('\033[92mAll tests passed!')

def optimize_test(target):
    w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
    expected_w = np.array([[-0.70916784], [-0.42390859]])
    expected_b = np.float64(2.26891346)
    expected_params = {"w": expected_w,
                       "b": expected_b}
   
    expected_dw = np.array([[0.06188603], [-0.01407361]])
    expected_db = np.float64(-0.04709353)
    expected_grads = {"dw": expected_dw,
                      "db": expected_db}
    
    expected_cost = [5.80154532, 0.31057104]
    expected_output = (expected_params, expected_grads, expected_cost)
    
    params, grads, costs = target(w, b, X, Y, num_iterations=101, learning_rate=0.1, print_cost=False)
    
    assert type(costs) == list, "Wrong type for costs. It must be a list"
    assert len(costs) == 2, f"Wrong length for costs. {len(costs)} != 2"
    assert np.allclose(costs, expected_cost), f"Wrong values for costs. {costs} != {expected_cost}"
    
    assert type(grads['dw']) == np.ndarray, f"Wrong type for grads['dw']. {type(grads['dw'])} != np.ndarray"
    assert grads['dw'].shape == w.shape, f"Wrong shape for grads['dw']. {grads['dw'].shape} != {w.shape}"
    assert np.allclose(grads['dw'], expected_dw), f"Wrong values for grads['dw']. {grads['dw']} != {expected_dw}"
    
    assert np.allclose(grads['db'], expected_db), f"Wrong values for grads['db']. {grads['db']} != {expected_db}"
    
    assert type(params['w']) == np.ndarray, f"Wrong type for params['w']. {type(params['w'])} != np.ndarray"
    assert params['w'].shape == w.shape, f"Wrong shape for params['w']. {params['w'].shape} != {w.shape}"
    assert np.allclose(params['w'], expected_w), f"Wrong values for params['w']. {params['w']} != {expected_w}"
    
    assert np.allclose(params['b'], expected_b), f"Wrong values for params['b']. {params['b']} != {expected_b}"

    
    print('\033[92mAll tests passed!')   
        
def predict_test(target):
    w = np.array([[0.3], [0.5], [-0.2]])
    b = -0.33333
    X = np.array([[1., -0.3, 1.5],[2, 0, 1], [0, -1.5, 2]])
    
    pred = target(w, b, X)
    
    assert type(pred) == np.ndarray, f"Wrong type for pred. {type(pred)} != np.ndarray"
    assert pred.shape == (1, X.shape[1]), f"Wrong shape for pred. {pred.shape} != {(1, X.shape[1])}"
    assert np.bitwise_not(np.allclose(pred, [[1., 1., 1]])), f"Perhaps you forget to add b in the calculation of A"
    assert np.allclose(pred, [[1., 0., 1]]), f"Wrong values for pred. {pred} != {[[1., 0., 1.]]}"
    
    print('\033[92mAll tests passed!')
    
def model_test(target):
    np.random.seed(0)
    
    expected_output = {'costs': [np.array(0.69314718)], 
                   'Y_prediction_test': np.array([[1., 1., 0.]]), 
                   'Y_prediction_train': np.array([[1., 1., 0., 1., 0., 0., 1.]]), 
                   'w': np.array([[ 0.08639757],
                           [-0.08231268],
                           [-0.11798927],
                           [ 0.12866053]]), 
                   'b': -0.03983236094816321}
    
    # Use 3 samples for training
    b, Y, X = 1.5, np.array([1, 0, 0, 1, 0, 0, 1]).reshape(1, 7), np.random.randn(4, 7),

    # Use 6 samples for testing
    x_test = np.random.randn(4, 3)
    y_test = np.array([0, 1, 0])

    d = target(X, Y, x_test, y_test, num_iterations=50, learning_rate=0.01)
    
    assert type(d['costs']) == list, f"Wrong type for d['costs']. {type(d['costs'])} != list"
    assert len(d['costs']) == 1, f"Wrong length for d['costs']. {len(d['costs'])} != 1"
    assert np.allclose(d['costs'], expected_output['costs']), f"Wrong values for d['costs']. {d['costs']} != {expected_output['costs']}"
    
    assert type(d['w']) == np.ndarray, f"Wrong type for d['w']. {type(d['w'])} != np.ndarray"
    assert d['w'].shape == (X.shape[0], 1), f"Wrong shape for d['w']. {d['w'].shape} != {(X.shape[0], 1)}"
    assert np.allclose(d['w'], expected_output['w']), f"Wrong values for d['w']. {d['w']} != {expected_output['w']}"
    
    assert np.allclose(d['b'], expected_output['b']), f"Wrong values for d['b']. {d['b']} != {expected_output['b']}"
    
    assert type(d['Y_prediction_test']) == np.ndarray, f"Wrong type for d['Y_prediction_test']. {type(d['Y_prediction_test'])} != np.ndarray"
    assert d['Y_prediction_test'].shape == (1, x_test.shape[1]), f"Wrong shape for d['Y_prediction_test']. {d['Y_prediction_test'].shape} != {(1, x_test.shape[1])}"
    assert np.allclose(d['Y_prediction_test'], expected_output['Y_prediction_test']), f"Wrong values for d['Y_prediction_test']. {d['Y_prediction_test']} != {expected_output['Y_prediction_test']}"
    
    assert type(d['Y_prediction_train']) == np.ndarray, f"Wrong type for d['Y_prediction_train']. {type(d['Y_prediction_train'])} != np.ndarray"
    assert d['Y_prediction_train'].shape == (1, X.shape[1]), f"Wrong shape for d['Y_prediction_train']. {d['Y_prediction_train'].shape} != {(1, X.shape[1])}"
    assert np.allclose(d['Y_prediction_train'], expected_output['Y_prediction_train']), f"Wrong values for d['Y_prediction_train']. {d['Y_prediction_train']} != {expected_output['Y_prediction_train']}"
    
    print('\033[92mAll tests passed!')
    
