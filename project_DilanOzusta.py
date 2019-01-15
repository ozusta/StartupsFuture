from operator import itemgetter
import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import Validator
from pybrain.structure import SigmoidLayer
import random
import os.path

# Column indexes

investment_column = 11
date_column =  10
name_column = 1
round_code_column = 9
status_column = 5

class Company:
    
    def __init__(self, list_of_round_rows, company_row):
        
        company_name = company_row[name_column]     
        self.Name = company_name
        
        status = company_row[status_column]
        
        # Changing the status to a numeric value
        if status == 'operating':
            self.Status = None
        elif status == 'ipo' or status == 'acquired':
            self.Status = 1
        else:
            self.Status = 0
        
        rnd_number = self.merge_rounds_together(list_of_round_rows)
        
        investments = [r[investment_column] for r in rnd_number]
        
        # Fills the rest of the indexes in investments list with 0 where there's no investment for the company
        if len(investments) > 10:
            investments = investments[:10]
        for i in range(len(investments), 10):
            investments.append(0)
        self.Investments = tuple(investments)
    
    def merge_rounds_together(self, list_of_round_rows):
        '''
        Given list of round rows, creates a list that includes the investment
        amounts for the company. Each round code needs only to be counted once,
        so if there's more than one same round, we'll add their investment 
        amounts together and return this list sorted by chronological order.
        Parameters:
            list_of_round_rows: a list that includes all investments of a company.
        Returns: a list
        '''
        rnd = {} 
        rnd_number = []
        
        for row in list_of_round_rows:
            round_code = row[round_code_column]
            if round_code != "":
                if round_code in rnd:
                    rnd[round_code][investment_column] += row[investment_column] 
                else:
                    rnd[round_code] = row
            else:
                rnd_number.append(row)
                
        rnd_number += rnd.values()
        rnd_number.sort(key=itemgetter(date_column))
        
        return rnd_number
                
    def __str__(self):
        return self.Name + " " + str(self.Investments) + " " + str(self.Status) 
        
def read_file(path):
    '''
    With a given filepath opens and reads the file
    Parameters:
        path: a filepath
    Returns: the file
    '''
    opened_file = open(path)
    return opened_file
    
def split_rows(opened_file):
    '''
    Split each row in the given file from the comma
    Parameters:
        opened_file: a file that has been read into the program
    Returns: a list
    '''
    
    result_list = [row.split(",") for row in opened_file.read().splitlines()]
    return result_list

def exclude_column_names(lst):
    '''
    From a given list, takes out its column names. 
    Parameters:
        lst: a list
    Returns: a list
    '''
    lst = lst[1:]

def isValidRow(round_row, name):
    '''
    Checks if the row is fit for consideration, data should exclude 
    improperly formatted data
    Parameters:
        round_row: list representing a row in "rounds.csv"
        name: name of the company
    Returns: boolean value
    '''
    return name == round_row[name_column] and len(round_row) == 12 and round_row[investment_column] != "" and int(round_row[investment_column]) != 0

def getCompanyList():
    '''
    Creates the company list with its name, investments and status information
    Parameters: None
    Returns: a list
    '''
    rounds_file = read_file("rounds.csv")
    companies_file = read_file("companies.csv")
    
    round_rows_list = split_rows(rounds_file)
    company_rows_list = split_rows(companies_file)
    
    assert type(round_rows_list) == list
    assert type(company_rows_list) == list
    
    exclude_column_names(round_rows_list)
    exclude_column_names(company_rows_list)
    rounds_file.close()
    companies_file.close()
    
    #Uncomment following line to test the code. It should give some results, but won't give good results!
    #company_rows_list = company_rows_list[:50]
    
    #Creates the company list
    print "Cleaning Data"
    i = 0
    comp_list = []
    for company_row in company_rows_list:
        cname = company_row[name_column]
        lst = []
        for round_row in round_rows_list:
            if isValidRow(round_row, cname):
                round_row[investment_column] = int(round_row[investment_column])
                lst.append(round_row)
        if len(lst) >= 1:
            c = Company(lst, company_row)
            assert len(c.Investments) == 10
            comp_list.append(c)
        i+=1
        if i%1000 == 0:
            print float(i)*100 / len(company_rows_list), "% complete"
    
    assert type(comp_list) == list
    
    for c in comp_list:
        assert isinstance(c, Company)
        
    return comp_list
    
def nn_vs_majority_plot(nn_accuracy_lst, majority):
    '''
    Creates a line plot for NN accuracy vs majority predictor.
    Parameters: 
        nn_accuracy_lst: the accuracy of NN over 10 tries
        majority: the accuracy of majority predictor
    Returns: None
    '''
    plt.clf()
    plt.plot(nn_accuracy_lst, label = "NN accuracy")
    plt.plot(majority, label = "Majority predictor")
    plt.xlim(0,9)
    plt.ylim(0.5,0.6)
    plt.xlabel("Number of tries")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.savefig("accuracy_comparison.png")
    
    assert os.path.exists("accuracy_comparison.png")
    
def write_output(filename, unclassified_data, net):
    '''
    Writes the output of unclassified data from the specified network to a file.
    Parameters:
        filename: name of the output file
        unclassified_data:  a list includes companies with investment and status 
        information where status is None
        net: name of the neural network
    Returns: None 
    '''
    out_file = open(filename,"w")
    for c in unclassified_data:
        result = net.activate(c.Investments)
        out_file.write(c.Name + " " + str(result[0]) + "\n")
    out_file.close()
    
    assert os.path.exists(filename)
    
def runNN(classified_data, split_point, unclassified_data = None):
    '''
    Splits the classified data into training and test by splitting it from split_point,
    creates an NN and trains it on the training data, tests the NN on test data and returns the accuracy
    and runs NN on the unclassified data if it exists and prints results to file
    Parameters:
        classified_data: a list includes companies with investment and status 
        information where status is either 0 or 1
        split_point: the point where the list will be divided into two
        classified_data: a list includes companies with investment and status 
        information where status is None
    Returns: a float
    
    '''
    random.shuffle(classified_data)
    training_data = classified_data[:split_point]
    test_data = classified_data[split_point:]
    
    # Creating the neural network, SigmoidLayer for keeping the results between 0 and 1
    
    net = buildNetwork(10, 10, 10, 1, bias=True, outclass=SigmoidLayer)
    
    # Training NN
    
    ds = SupervisedDataSet(10, 1) # Create empty training dataset for NN
    
    for data in training_data: # fill NN
        ds.addSample(data.Investments, (data.Status,))
    
    trainer = BackpropTrainer(net, ds) # train the NN with training dataset
    trainer.trainUntilConvergence()
    
    sum_accuracy = 0
    
    for c in test_data:
        result = net.activate(c.Investments) # runs NN with Investments
        result = result[0]
        sum_accuracy += (1 - abs(result - c.Status))
    accuracy_nn = sum_accuracy / len(test_data)
    
    assert accuracy_nn >= 0 and accuracy_nn <= 1
    
    if unclassified_data != None:
        write_output("output.txt", unclassified_data, net)
    return accuracy_nn

def get_majority(classified_data):
    '''
    Finds the accuracy of majority predictor
    Parameters:
        classified_data: a list includes companies with investment and status 
        information where status is either 0 or 1
    Returns: a float
    '''
    count_1 = 0
    for i in classified_data:
        if i.Status == 1:
            count_1 += 1
    majority = max(count_1,len(classified_data) - count_1) / float(len(classified_data))
    
    assert type(majority) == float
    assert majority >= 0 and majority <= 1 
    
    return majority
    
def getAccuracyList(classified_data, unclassified_data, split_point):
    '''
    Creates a list that includes the accuracies of the neural network
    Parameters:
        classified_data: a list includes companies with investment and status 
        information where status is either 0 or 1
        unclassified_data: a list includes companies with investment and status 
        information where status is None
        split_point: the point where the classified data list will be divided into two
    Returns: a list of floats
    '''
    
    nn_accuracy_lst = []
    print "Training NN 10 times"
    for i in range(10):
        print i*10, "% complete"
        if i == 0:
            accuracy_nn = runNN(classified_data, split_point, unclassified_data)
        else:
            accuracy_nn = runNN(classified_data, split_point)           
        nn_accuracy_lst.append(accuracy_nn)
        
    return nn_accuracy_lst
        
def main():
    companies = getCompanyList()
    
    # Splitting the dataset: Unclassified and Classified (and training and test)
    
    unclassified_data = []
    classified_data = []
    
    for company in companies:
        if company.Status == None:
            unclassified_data.append(company)
        else:
            classified_data.append(company)
            
    split_point = len(classified_data) * 4/5
        
    majority = get_majority(classified_data)
    nn_accuracy_lst = getAccuracyList(classified_data, unclassified_data, split_point)
        
    print "Test data length:", len(classified_data) - split_point
    print "Training data length:", split_point
    print "Unclassified data length:", len(unclassified_data)
    print "Accuracy of majority predictor:", majority
    print "Accuracy of NN over 10 tries:", nn_accuracy_lst
    
    nn_vs_majority_plot(nn_accuracy_lst, [majority]*10)
    
if __name__ == "__main__":
    main()