#Functions to be used

from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import numpy as np
import qiskit.quantum_info as qi
from math import pi
import math
import matplotlib.pylab as plt
from itertools import product
from qiskit.quantum_info import Statevector, Operator,  DensityMatrix,state_fidelity
import itertools
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from tqdm.auto import tqdm
#from tqdm import tqdm
import time
import warnings
from smt.sampling_methods import LHS
warnings.filterwarnings("ignore")
import torch
import gpytorch
import json
import uuid
from botorch.models.gpytorch import GPyTorchModel
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.posteriors import GPyTorchPosterior




# Function to calculate fidelity
def calculate_fidelity(real_state, pred_state):
    return qi.state_fidelity(real_state, pred_state)



def get_expectation_values(circuit, thetas,  initial_state, target_state):
    
    circ = circuit(thetas, initial_state)
    n_qubits = circ.num_qubits

    #Operator to use

    coefficients,_ = get_coefficients(target_state, n_qubits)
    operators = [ Operator.from_label(i) for i in coefficients.keys()]

    simulator_contr = Aer.get_backend('statevector_simulator')
    compiled_circuit_contr = transpile(circ, simulator_contr)
    result_contr = simulator_contr.run(assemble(compiled_circuit_contr)).result()
    state_vector = result_contr.get_statevector()

    # Calculate fidelity
    real_fidelity = calculate_fidelity(target_state, state_vector.data)

    unnormalized_state_vector = state_vector.data*np.linalg.norm(state_vector)
    expectation_values = []
    for op in operators:
        expectation_value = (np.dot(np.conj(unnormalized_state_vector), np.dot(op.data, unnormalized_state_vector))).real
        expectation_values.append(expectation_value)   

    return np.array(expectation_values), real_fidelity, coefficients


def get_new_samples(n_samples, n_thetas):
    XNewOut = []
    for _ in range(n_samples):
        thetas_new = np.random.uniform(0,2*pi, n_thetas)
        XNewOut.append(thetas_new)
    return np.array(XNewOut)




def get_real_samples(circuit, n_samples, n_thetas, initial_state, target_state, type_sampling = 'Random_Uniform'):

    if type_sampling=='Random_Uniform':
        X_out = get_new_samples(n_samples, n_thetas)
    elif type_sampling == 'LHS':
        X_out = get_new_samples_lhs(n_samples, n_thetas)
    else:
      raise Exception("Sorry, Sampling method unavailable")         
    
    fidelities_real = []
    expectations_values = []
    for j in range(n_samples):
        thetas_new = X_out[j]
        expectation_values, fid_real, _ = get_expectation_values(circuit, thetas_new[np.newaxis,:], initial_state, target_state)
        expectations_values.append(expectation_values)
        fidelities_real.append(fid_real)

    expectations_values =  np.array(expectations_values)
    fidelities_real = np.array(fidelities_real)

    return X_out, np.real(expectations_values), fidelities_real




def expectation2fidelity(expectations,coefficients_array, num_qubits):
    #Calculate the Fidelity using the observables       
    fid = np.abs(np.sum((np.hstack((np.ones((np.shape(expectations)[0],1)), expectations *coefficients_array))/(2**num_qubits)), axis=1))    

    return fid




    
def get_new_samples_lhs(n_samples, n_thetas):
    xlimits = np.array([[0,2*pi]]*n_thetas )
    sampling = LHS(xlimits=xlimits)
    XNewOut = sampling(n_samples)

    return XNewOut

def expected_improvement(estimated_expectation_values, std_estimated_expectation_values, y_best_fidelity):

    sigma = std_estimated_expectation_values

    imp = (estimated_expectation_values  - y_best_fidelity)
    z = imp/sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    return ei


def ucb(estimated_expectation_values, std_estimated_expectation_values, kappa=2.0):
    ucb_value = estimated_expectation_values + kappa * std_estimated_expectation_values
    return ucb_value


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) #gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)) #
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)





def train_surrogate_models_gp(args):

        X_train, y_train, X_test, y_test, params = args

        if len(X_test)!=0: test=True
        else: test=False
        
        if params.get('scaler_target', None)=='StandardScaler': #Normalizate
                scaler_target = StandardScaler()
                #scaler_target = params.get('scaler_target', None)
                scaler_target = scaler_target.fit(y_train[:,np.newaxis])
                y_train = scaler_target.transform(y_train[:,np.newaxis]).flatten()
                if test:
                        y_test = scaler_target.transform(y_test[:,np.newaxis]).flatten()
        else:
                scaler_target = None

        X_train_tensor, y_train_tensor = torch.tensor(X_train).double(), torch.tensor(y_train).double()
        X_test_tensor, y_test_tensor = torch.tensor(X_test).double(), torch.tensor(y_test).double()

        verbose = params.get('verbose', True)

        training_iter = params.get('training_iter', 20)
        lr = params.get('learning_rate', 0.1)
        initial_length_scale = params.get('initial_lengthscale', 1.0)
        initial_noise_likelihood = params.get('initial_noise', 0.1) 
        initial_output_scale = params.get('initial_output_scale', 0.1) 

        #gpytorch.settings.max_cholesky_size(1.0e-2) 
        #gpytorch.settings.cholesky_jitter(1.0e-2) 

        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-12))
        
    
        
        model = ExactGPModel(X_train_tensor, y_train_tensor, likelihood)
        model = model.double()

        model.covar_module.base_kernel.lengthscale = torch.tensor(initial_length_scale)
        model.likelihood.noise =  torch.tensor(initial_noise_likelihood)
        model.covar_module.outputscale = torch.tensor(initial_output_scale)


        gpytorch.settings.max_cholesky_size(1.0e-2) 
        gpytorch.settings.cholesky_jitter(1.0e-2) 

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
      
        def closure(i):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(X_train_tensor)
                # Calc loss and backprop gradients
                loss = -mll(output, y_train_tensor)
                loss.backward()
                if verbose:
                    print('Iter %d/%d - Loss: %.3f  outputscale: %.3f   lengthscale: %.3f   noise: %.5f' % (
                            i + 1, training_iter, loss.item(),
                            model.covar_module.outputscale.item(),
                            model.covar_module.base_kernel.lengthscale.item(),
                            model.likelihood.noise.item()))
            
                return loss


        metrics = {'loss_train':[], 'loss_test':[],
                   'MSE_train':[], 'MSE_test':[]}
        for i in range(training_iter):
                
                model.train()
                likelihood.train()
                optimizer.step(lambda: closure(i))

                model.eval()
                likelihood.eval()       
                with torch.no_grad(), gpytorch.settings.cholesky_jitter(1.0e-2), gpytorch.settings.max_preconditioner_size(10), \
                            gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
                                                

                        y_pred_train = model(X_train_tensor)
                        covar = y_pred_train.lazy_covariance_matrix


                        train_loss = -mll(y_pred_train, y_train_tensor)
                        metrics['loss_train'].append(train_loss.item())

                        if params.get('scaler_target', None) != None:
                                y_pred_train = (scaler_target.inverse_transform(y_pred_train.mean.numpy()[:,np.newaxis])).flatten()
                                y_train_tensor_numpy = (scaler_target.inverse_transform(y_train_tensor[:,np.newaxis])).flatten()
                        else:
                                y_pred_train = y_pred_train.mean.numpy()
                                y_train_tensor_numpy =  y_train_tensor.numpy()

                        mse_train = np.mean((y_pred_train-y_train_tensor_numpy)**2)
                        metrics['MSE_train'].append(mse_train)

                        if test:
                                y_pred_test = model(X_test_tensor)
                                test_loss = -mll(y_pred_test, y_test_tensor)                                
                                if params.get('scaler_target', None) != None:
                                        y_pred_test = (scaler_target.inverse_transform(y_pred_test.mean.numpy()[:,np.newaxis])).flatten()
                                        y_test_tensor_numpy = (scaler_target.inverse_transform(y_test_tensor[:,np.newaxis])).flatten()
                                else:
                                        y_pred_test = y_pred_test.mean.numpy()
                                        y_test_tensor_numpy =  y_test_tensor.numpy()


                                mse_test = np.mean((y_pred_test - y_test_tensor_numpy)**2)
                                metrics['MSE_test'].append(mse_test)
                                metrics['loss_test'].append(test_loss.item())

        return model, likelihood, metrics, scaler_target




def predict_surrogate_models_gp(model, likelihood,  X_test, params_predict):
        X_test_tensor = torch.tensor(X_test)
        model.eval()    
        likelihood.eval() 
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1.0e-2), gpytorch.settings.max_preconditioner_size(10), \
                gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False) : #, gpytorch.settings.fast_pred_var(solves=False): 
            

            y_pred_test = likelihood(model(X_test_tensor)  )
            covar = y_pred_test.lazy_covariance_matrix

            if params_predict.get('scaler_target', None)!=None:
                scaler_target = params_predict.get('scaler_target', None)
                mean_pred = (scaler_target.inverse_transform(y_pred_test.mean.numpy()[:, np.newaxis])).flatten()
                std_pred = np.array(np.sqrt(y_pred_test.variance)*scaler_target.scale_).flatten()
           

            else:
                mean_pred = y_pred_test.mean.numpy()
                std_pred = y_pred_test.stddev.numpy() 



        return mean_pred, std_pred, y_pred_test
  


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        elif isinstance(obj, uuid.UUID):
            return str(obj)  # Convert UUID to string
        elif isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        elif isinstance(obj, np.float32):
             return float(obj)
        
        return json.JSONEncoder.default(self, obj)
            

def get_coefficients(target_state, n_qubits):

    str_list = [''.join(item) for item in product(['I','X','Y', 'Z'], repeat=n_qubits)]
    operators = [ Operator.from_label(i) for i in str_list]
    #target_state = np.array([1,0,0,1])/np.sqrt(2)

    all_coef, coef = {}, {}
    for i in range(len(operators)):
        val = (np.dot(np.conj(target_state), np.dot(operators[i],target_state))).real
        all_coef[str_list[i]] = val

        if np.abs(val)>1e-8:
            coef[str_list[i]] = val

    del coef['I'*n_qubits] #Remove the Identity

    return coef, n_qubits




def get_fidelity_from_expectations_values(surrogate_models,X_data, coefficients_array, num_qubits):
        n_samples_predict, n_surrogates_models = X_data.shape[0], len(surrogate_models['gp_model'])

        test_mean_estimated_expectation = np.zeros((n_samples_predict, n_surrogates_models))
        test_std_estimated_expectation = np.zeros((n_samples_predict, n_surrogates_models))
        for n_sm in range(n_surrogates_models):
                gp_model = surrogate_models['gp_model'][n_sm]
                ll = surrogate_models['likelihood'][n_sm]
                params_predict = {'scaler_target':surrogate_models['scaler_target'][n_sm]}
                test_mean_estimated_expectation[:,n_sm], test_std_estimated_expectation[:,n_sm], _ = predict_surrogate_models_gp(gp_model, ll, X_data, params_predict)

        mean_estimated_fidelities =  expectation2fidelity(test_mean_estimated_expectation, coefficients_array, num_qubits)
        std_estimated_fidelities = np.sqrt(np.sum(((coefficients_array/(2**num_qubits))**2) *(test_std_estimated_expectation**2), 1))

        return mean_estimated_fidelities, std_estimated_fidelities




def expectation2eigenvalue(expectations,coefficients_array, num_qubits):
    #Calculate the Eigenvalue using the observables       
    eigenvalue = np.sum( expectations *coefficients_array, axis=1)  
    #eigenvalue = np.sum(((expectations *coefficients_array)/(2**num_qubits)), axis=1)


    return eigenvalue


def get_eigenvalues_from_expectations_values(surrogate_models,X_data, coefficients_array, num_qubits):
        n_samples_predict, n_surrogates_models = X_data.shape[0], len(surrogate_models['gp_model'])

        test_mean_estimated_expectation = np.zeros((n_samples_predict, n_surrogates_models))
        test_std_estimated_expectation = np.zeros((n_samples_predict, n_surrogates_models))
        for n_sm in range(n_surrogates_models):
                gp_model = surrogate_models['gp_model'][n_sm]
                ll = surrogate_models['likelihood'][n_sm]
                params_predict = {'scaler_target':surrogate_models['scaler_target'][n_sm]}
                test_mean_estimated_expectation[:,n_sm], test_std_estimated_expectation[:,n_sm], _ = predict_surrogate_models_gp(gp_model, ll, X_data, params_predict)

        mean_estimated_eigenvalues =  expectation2eigenvalue(test_mean_estimated_expectation, coefficients_array, num_qubits)
        std_estimated_eigenvalues = np.sqrt(np.sum(((coefficients_array)**2) *(test_std_estimated_expectation**2), 1))

        #mean_estimated_eigenvalues =  expectation2eigenvalue(test_mean_estimated_expectation, coefficients_array, num_qubits)
        #std_estimated_eigenvalues = np.sqrt(np.sum(((coefficients_array/(2**num_qubits))**2) *(test_std_estimated_expectation**2), 1))

        return mean_estimated_eigenvalues, std_estimated_eigenvalues


class WrappedGPyTorchModelMultiGPFidelity(GPyTorchModel):
    def __init__(self, surrogate_models, coefficients_array, num_qubits ):
        super(WrappedGPyTorchModelMultiGPFidelity, self).__init__()

        self.surrogate_models = surrogate_models
        self.coefficients_array = coefficients_array
        self.num_qubits = num_qubits


    def posterior(self, X, observation_noise=False, **kwargs):


        self.mean_fid, self.covar_fid = self.get_fidelity_from_expectations_values_withGradients(X)
        dist = gpytorch.distributions.MultivariateNormal(self.mean_fid, self.covar_fid)
        return GPyTorchPosterior(dist)


    def predict_surrogate_models_gp_with_gradients(self, model, ll, params_predict, X_data):
        model.eval()
        ll.eval()

        with gpytorch.settings.cholesky_jitter(1.0e-2), gpytorch.settings.max_preconditioner_size(10), \
                        gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False) :
      
            y_pred_test = ll(model(X_data))

            if params_predict.get('scaler_target', None)!=None:
                scaler_target = params_predict.get('scaler_target', None)

                mean_scaler = torch.tensor(scaler_target.mean_, dtype=torch.float64)
                scale_scaler = torch.tensor(scaler_target.scale_, dtype=torch.float64)

                #unscaling
                y_mean_pred_unscaled = y_pred_test.mean *scale_scaler + mean_scaler
                y_std_pred_unscaled = torch.sqrt(y_pred_test.variance) * scale_scaler
                y_covar_pred_unscaled = y_pred_test.covariance_matrix/(scale_scaler**2)
            else:
                y_mean_pred_unscaled = y_pred_test.mean 
                y_covar_pred_unscaled = y_pred_test.covariance_matrix

        return y_mean_pred_unscaled, y_covar_pred_unscaled


    #Aqui sacare el add
    def get_fidelity_from_expectations_values_withGradients(self, X_data):

            n_surrogates_models = len(self.surrogate_models['gp_model'])
            mean_estimated_expectation, covar_estimated_expectation = [],[]
            for n_sm in range(n_surrogates_models):
                gp_model = self.surrogate_models['gp_model'][n_sm]
                ll = self.surrogate_models['likelihood'][n_sm]         
                params_predict = {'scaler_target':self.surrogate_models['scaler_target'][n_sm]}
                y_mean0,  y_covar0 =  self.predict_surrogate_models_gp_with_gradients(gp_model, ll, params_predict, X_data)

                mean_estimated_expectation.append(y_mean0)
                covar_estimated_expectation.append(y_covar0)

            mean_estimated_expectation = (torch.stack(mean_estimated_expectation)).squeeze(-1)
            covar_estimated_expectation = (torch.stack(covar_estimated_expectation)).squeeze(-1)

            mean_estimated_fidelities = (self.expectation2fidelity_with_gradients(mean_estimated_expectation)).unsqueeze(1)
  
            coefficients_array_expanded = torch.tensor(self.coefficients_array).unsqueeze(1).unsqueeze(2)
            covar_estimated_fidelities_coefficients = ((covar_estimated_expectation**2) * ((coefficients_array_expanded/(2**self.num_qubits))**2))
            final_covar_estimated_fidelities = (torch.sqrt(torch.sum(covar_estimated_fidelities_coefficients,0))).unsqueeze(1)

            return mean_estimated_fidelities, final_covar_estimated_fidelities


    def expectation2fidelity_with_gradients(self, mean_estimated_expectation):

        elementwise_product = torch.transpose(mean_estimated_expectation,0,1 ) * torch.tensor(self.coefficients_array)
        ones_column = torch.ones(mean_estimated_expectation.shape[1],1)
        stacked = torch.cat((ones_column, elementwise_product), dim=1)
        divided = stacked / (2**self.num_qubits)
        summed = torch.sum(divided, dim=1)
        fid = torch.abs(summed)

        return fid
    @property
    def num_outputs(self):
        return 1


    




def get_expectation_values_hamiltonian(circuit, thetas, initial_state, H):
   
    circ = circuit(thetas, initial_state)
    statevector = Statevector.from_instruction(circ)
    expectation_values = []
    for pauli_op in H:
        coefficient = pauli_op.coeff
        observable = pauli_op.to_matrix()
        expectation = coefficient * statevector.expectation_value(observable)
        expectation_values.append( expectation.real )
    expectation_values = -np.array(expectation_values) #Minus because the code was made to maximise, with the minus the code minimise
    return expectation_values, np.sum(expectation_values)


def get_real_samples_vqe(circuit, n_samples, n_thetas, initial_state, H, type_sampling = 'Random_Uniform'):
   
    if type_sampling=='Random_Uniform':
        X_out = get_new_samples(n_samples, n_thetas)
    elif type_sampling == 'LHS':
        X_out = get_new_samples_lhs(n_samples, n_thetas)
    else:
      raise Exception("Sorry, Sampling method unavailable")   

    expectations_values = []
    for j in range(n_samples):
        thetas_new = X_out[j]
        expectation_values, _ = get_expectation_values_hamiltonian(circuit, thetas_new[np.newaxis,:], initial_state, H)
        expectations_values.append(expectation_values)

    expectations_values =  np.array(expectations_values)

    return X_out, np.real(expectations_values), np.sum(expectations_values,1)   








class WrappedGPyTorchModelMultiGPHamiltonian(GPyTorchModel):
    def __init__(self, surrogate_models, coefficients_array, num_qubits ):
        super(WrappedGPyTorchModelMultiGPHamiltonian, self).__init__()

        self.surrogate_models = surrogate_models
        self.coefficients_array = coefficients_array
        self.num_qubits = num_qubits


    def posterior(self, X, observation_noise=False, **kwargs):


        self.mean_fid, self.covar_fid = self.get_eigenvalues_from_expectations_values_withGradients(X)
        dist = gpytorch.distributions.MultivariateNormal(self.mean_fid, self.covar_fid)
        return GPyTorchPosterior(dist)


    def predict_surrogate_models_gp_with_gradients(self, model, ll, params_predict, X_data):
        model.eval()
        ll.eval()

        with gpytorch.settings.cholesky_jitter(1.0e-2), gpytorch.settings.max_preconditioner_size(10), \
                        gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False) :
      
            y_pred_test = ll(model(X_data))

            if params_predict.get('scaler_target', None)!=None:
                scaler_target = params_predict.get('scaler_target', None)

                mean_scaler = torch.tensor(scaler_target.mean_, dtype=torch.float64)
                scale_scaler = torch.tensor(scaler_target.scale_, dtype=torch.float64)

                #unscaling
                y_mean_pred_unscaled = y_pred_test.mean *scale_scaler + mean_scaler
                y_std_pred_unscaled = torch.sqrt(y_pred_test.variance) * scale_scaler
                y_covar_pred_unscaled = y_pred_test.covariance_matrix/(scale_scaler**2)
            else:
                y_mean_pred_unscaled = y_pred_test.mean 
                y_covar_pred_unscaled = y_pred_test.covariance_matrix

        return y_mean_pred_unscaled, y_covar_pred_unscaled


    #Aqui sacare el add
    def get_eigenvalues_from_expectations_values_withGradients(self, X_data):

            n_surrogates_models = len(self.surrogate_models['gp_model'])
            mean_estimated_expectation, covar_estimated_expectation = [],[]
            for n_sm in range(n_surrogates_models):
                gp_model = self.surrogate_models['gp_model'][n_sm]
                ll = self.surrogate_models['likelihood'][n_sm]         
                params_predict = {'scaler_target':self.surrogate_models['scaler_target'][n_sm]}
                y_mean0,  y_covar0 =  self.predict_surrogate_models_gp_with_gradients(gp_model, ll, params_predict, X_data)

                mean_estimated_expectation.append(y_mean0)
                covar_estimated_expectation.append(y_covar0)

            mean_estimated_expectation = (torch.stack(mean_estimated_expectation)).squeeze(-1)
            covar_estimated_expectation = (torch.stack(covar_estimated_expectation)).squeeze(-1)

            mean_estimated_eigenvalues = (self.expectation2eigenvalue_with_gradients(mean_estimated_expectation)).unsqueeze(1)
  
            coefficients_array_expanded = torch.tensor(self.coefficients_array).unsqueeze(1).unsqueeze(2)
            covar_estimated_eigenvalues_coefficients = ((covar_estimated_expectation**2) * ((coefficients_array_expanded)**2))
            #covar_estimated_eigenvalues_coefficients = ((covar_estimated_expectation**2) * ((coefficients_array_expanded/(2**num_qubits))**2))

            final_covar_estimated_eigenvalues = (torch.sqrt(torch.sum(covar_estimated_eigenvalues_coefficients,0))).unsqueeze(1)

            return mean_estimated_eigenvalues, final_covar_estimated_eigenvalues


    def expectation2eigenvalue_with_gradients(self, mean_estimated_expectation):

        elementwise_product = torch.transpose(mean_estimated_expectation,0,1 ) * torch.tensor(self.coefficients_array)
        
        #divided = elementwise_product / (2**self.num_qubits)
        #summed = torch.sum(divided, dim=1)
        summed = torch.sum(elementwise_product, dim=1)
        #eigenvalue = torch.abs(summed)
        eigenvalue = summed

        return eigenvalue
    

    @property
    def num_outputs(self):
        return 1



class WrappedGPyTorchModelMonoGP(GPyTorchModel):
    def __init__(self, gp_model_original):
        super(WrappedGPyTorchModelMonoGP, self).__init__()
        self.gp_model_original = gp_model_original

    def posterior(self, X, observation_noise=False, **kwargs):
        self.gp_model_original.eval()
        #with torch.no_grad():
        dist = self.gp_model_original(X)
        return GPyTorchPosterior(dist)
    
    @property
    def num_outputs(self):
        return 1











