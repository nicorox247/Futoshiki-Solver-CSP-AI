"""
Each futoshiki board is represented as a dictionary with string keys and
int values.
e.g. my_board['A1'] = 8

Empty values in the board are represented by 0

An * after the letter indicates the inequality between the row represented
by the letter and the next row.
e.g. my_board['A*1'] = '<' 
means the value at A1 must be less than the value
at B1

Similarly, an * after the number indicates the inequality between the
column represented by the number and the next column.
e.g. my_board['A1*'] = '>' 
means the value at A1 is greater than the value
at A2

Empty inequalities in the board are represented as '-'

"""
import sys

#======================================================================#
#*#*#*# Optional: Import any allowed libraries you may need here #*#*#*#
#======================================================================#
import numpy as np
import time
import copy
#=================================#
#*#*#*# Your code ends here #*#*#*#
#=================================#

ROW = "ABCDEFGHI"
COL = "123456789"

class Board:
    '''
    Class to represent a board, including its configuration, dimensions, and domains
    '''
    
    def get_board_dim(self, str_len):
        '''
        Returns the side length of the board given a particular input string length
        '''
        d = 4 + 12 * str_len
        n = (2+np.sqrt(4+12*str_len))/6
        if(int(n) != n):
            raise Exception("Invalid configuration string length")
        
        return int(n)
        
    def get_config_str(self):
        '''
        Returns the configuration string
        '''
        return self.config_str
        
    def get_config(self):
        '''
        Returns the configuration dictionary
        '''
        return self.config
        
    def get_variables(self):
        '''
        Returns a list containing the names of all variables in the futoshiki board
        '''
        variables = []
        for i in range(0, self.n):
            for j in range(0, self.n):
                variables.append(ROW[i] + COL[j])
        return variables
    
    def convert_string_to_dict(self, config_string):
        '''
        Parses an input configuration string, retuns a dictionary to represent the board configuration
        as described above
        '''
        config_dict = {}
        
        for i in range(0, self.n):
            for j in range(0, self.n):
                cur = config_string[0]
                config_string = config_string[1:]
                
                config_dict[ROW[i] + COL[j]] = int(cur)
                
                if(j != self.n - 1):
                    cur = config_string[0]
                    config_string = config_string[1:]
                    config_dict[ROW[i] + COL[j] + '*'] = cur
                    
            if(i != self.n - 1):
                for j in range(0, self.n):
                    cur = config_string[0]
                    config_string = config_string[1:]
                    config_dict[ROW[i] + '*' + COL[j]] = cur
                    
        return config_dict
        
    def print_board(self):
        '''
        Prints the current board to stdout
        '''
        config_dict = self.config
        for i in range(0, self.n):
            for j in range(0, self.n):
                cur = config_dict[ROW[i] + COL[j]]
                if(cur == 0):
                    print('_', end=' ')
                else:
                    print(str(cur), end=' ')
                
                if(j != self.n - 1):
                    cur = config_dict[ROW[i] + COL[j] + '*']
                    if(cur == '-'):
                        print(' ', end=' ')
                    else:
                        print(cur, end=' ')
            print('')
            if(i != self.n - 1):
                for j in range(0, self.n):
                    cur = config_dict[ROW[i] + '*' + COL[j]]
                    if(cur == '-'):
                        print(' ', end='   ')
                    else:
                        print(cur, end='   ')
            print('')
    
    def __init__(self, config_string):
        '''
        Initialising the board
        '''
        self.config_str = config_string
        self.n = self.get_board_dim(len(config_string))
        if(self.n > 9):
            raise Exception("Board too big")
            
        self.config = self.convert_string_to_dict(config_string)
        self.domains = self.reset_domains()
        
        self.forward_checking(self.get_variables())
        
        
    def __str__(self):
        '''
        Returns a string displaying the board in a visual format. Same format as print_board()
        '''
        output = ''
        config_dict = self.config
        for i in range(0, self.n):
            for j in range(0, self.n):
                cur = config_dict[ROW[i] + COL[j]]
                if(cur == 0):
                    output += '_ '
                else:
                    output += str(cur)+ ' '
                
                if(j != self.n - 1):
                    cur = config_dict[ROW[i] + COL[j] + '*']
                    if(cur == '-'):
                        output += '  '
                    else:
                        output += cur + ' '
            output += '\n'
            if(i != self.n - 1):
                for j in range(0, self.n):
                    cur = config_dict[ROW[i] + '*' + COL[j]]
                    if(cur == '-'):
                        output += '    '
                    else:
                        output += cur + '   '
            output += '\n'
        return output
        
    def reset_domains(self):
        '''
        Resets the domains of the board assuming no enforcement of constraints
        '''
        domains = {}
        variables = self.get_variables()
        for var in variables:
            if(self.config[var] == 0):
                domains[var] = [i for i in range(1,self.n+1)]
            else:
                domains[var] = [self.config[var]]
                
        self.domains = domains
                
        return domains

    def forward_checking(self, reassigned_variables):
        '''
        Runs the forward checking algorithm to restrict the domains of all variables based on the values
        of reassigned variables
        '''
        #======================================================================#
		#*#*#*# TODO: Write your implementation of forward checking here #*#*#*#
		#======================================================================# 

        for variable in reassigned_variables:
            assigned_value = self.config[variable]
            
            # Skip if variable is not assigned (0 means unassigned)
            if assigned_value == 0:
                continue
            
            row, col = ROW.index(variable[0]), COL.index(variable[1])
            
            # Remove assigned value from all other cells in the same row
            for j in range(self.n):
                neighbor = ROW[row] + COL[j]
                if neighbor != variable and assigned_value in self.domains[neighbor]:
                    self.domains[neighbor].remove(assigned_value)
                    # If domain becomes empty, forward checking fails
                    if not self.domains[neighbor]:
                        return False
            
            # Remove assigned value from all other cells in the same column
            for i in range(self.n):
                neighbor = ROW[i] + COL[col]
                if neighbor != variable and assigned_value in self.domains[neighbor]:
                    self.domains[neighbor].remove(assigned_value)
                    # If domain becomes empty, forward checking fails
                    if not self.domains[neighbor]:
                        return False
            
            # Handle horizontal inequalities
            if col < self.n - 1:
                inequality_key = variable + '*'
                if inequality_key in self.config and self.config[inequality_key] in ['<', '>']:
                    neighbor = ROW[row] + COL[col + 1]
                    self.apply_inequality_forward_check(variable, neighbor, self.config[inequality_key])
            
            # Handle vertical inequalities
            if row < self.n - 1:
                inequality_key = ROW[row] + '*' + COL[col]
                if inequality_key in self.config and self.config[inequality_key] in ['<', '>']:
                    neighbor = ROW[row + 1] + COL[col]
                    self.apply_inequality_forward_check(variable, neighbor, self.config[inequality_key])

        return True
        #=================================#
		#*#*#*# Your code ends here #*#*#*#
		#=================================#

    def apply_inequality_forward_check(self, variable1, variable2, inequality):
        '''
        forward checking for neighboring cells based on inequality constraints
        '''
        domain1 = self.domains[variable1]
        domain2 = self.domains[variable2]

        if inequality == '<':
            # Update domain1: values less than the max of domain2
            max_domain2 = max(domain2) if domain2 else 0
            self.domains[variable1] = [val1 for val1 in domain1 if val1 < max_domain2]
            # Update domain2: values greater than the min of domain1
            min_domain1 = min(self.domains[variable1]) if self.domains[variable1] else self.n + 1
            self.domains[variable2] = [val2 for val2 in domain2 if val2 > min_domain1]
        elif inequality == '>':
            # Update domain1: values greater than the min of domain2
            min_domain2 = min(domain2) if domain2 else self.n + 1
            self.domains[variable1] = [val1 for val1 in domain1 if val1 > min_domain2]
            # Update domain2: values less than the max of domain1
            max_domain1 = max(self.domains[variable1]) if self.domains[variable1] else 0
            self.domains[variable2] = [val2 for val2 in domain2 if val2 < max_domain1]
        
    #=================================================================================#
	#*#*#*# Optional: Write any other functions you may need in the Board Class #*#*#*#
	#=================================================================================#
    
        #Functions that check constraints
    def row_constraint(self):
        '''returns false if board violated numbered row constraint'''
        for i in range(self.n):
            seen = set()
            for j in range(self.n):
                value = self.config[ROW[i] + COL[j]]
                if value != 0:
                    if value in seen:
                        return False
                    seen.add(value)

        return True

    def column_constraint(self):
        '''returns false if board violated numbered column constriant'''
        for j in range(self.n):
            seen = set()
            for i in range(self.n):
                value = self.config[ROW[i] + COL[j]]
                if value != 0:
                    if value in seen:
                        return False
                    seen.add(value)

        return True

    def binary_constraint(self):
        '''
        Checks that all inequality constraints are satisfied.
        Returns False if any inequality constraints are violated.
        '''
        for key, value in self.config.items():
            if '*' in key:  # This indicates an inequality constraint
                cell1, cell2 = self.get_cells_from_key(key)
                if value == '<' and self.config[cell1] != 0 and self.config[cell2] != 0:
                    if not self.config[cell1] < self.config[cell2]:
                        return False
                elif value == '>' and self.config[cell1] != 0 and self.config[cell2] != 0:
                    if not self.config[cell1] > self.config[cell2]:
                        return False
        return True
    
    def get_cells_from_key(self, key):
        '''
        Helper func to return the two cells involved in an inequality based on the key.
        '''
        if '*' in key:
            if key[-1] == '*':  # Horizontal inequality
                cell1 = key[:-1]
                row, col = cell1[0], cell1[1]
                cell2 = row + str(int(col) + 1)
            else:  # Vertical inequality
                row, col = key.split('*')
                cell1 = row + col
                next_row = ROW[ROW.index(row) + 1]
                cell2 = next_row + col
            return cell1, cell2
        return None, None
    #=================================#
	#*#*#*# Your code ends here #*#*#*#
	#=================================#

#================================================================================#
#*#*#*# Optional: You may write helper functions in this space if required #*#*#*#
#================================================================================# 

#=================================#
#*#*#*# Your code ends here #*#*#*#
#=================================#

def backtracking(board):
    '''
    Performs the backtracking algorithm to solve the board
    Returns only a solved board
    '''
    #==========================================================#
	#*#*#*# TODO: Write your backtracking algorithm here #*#*#*#
	#==========================================================#
    # Step 1: Check if all variables are assigned (base case)
    unassigned_vars = [var for var in board.get_variables() if board.config[var] == 0]
    if not unassigned_vars:
        # If there are no unassigned variables, return the board as solved
        return board

    # Step 2: Select variable with Minimum Remaining Value (MRV)
    mrv_var = min(unassigned_vars, key=lambda var: len(board.domains[var]))

    # Step 3: Try each value in the domain of the selected variable
    original_domain = board.domains[mrv_var][:]
    for value in original_domain:
        # Assign the value to the variable
        board.config[mrv_var] = value

        # Perform forward checking after the assignment
        domains_backup = copy.deepcopy(board.domains)  # Backup domains before applying forward checking
        if board.forward_checking([mrv_var]):
            # Recursively call backtracking to solve the rest of the board
            result = backtracking(board)
            if result:
                return result  # If a solution is found, return the solved board

        # Backtrack: undo the assignment and restore domains
        board.config[mrv_var] = 0
        board.domains = domains_backup

    # If no valid assignment is found, return None to indicate failure
    return None

    #=================================#
	#*#*#*# Your code ends here #*#*#*#
	#=================================#
    
def solve_board(board):
    '''
    Runs the backtrack helper and times its performance.
    Returns the solved board and the runtime
    '''
    #================================================================#
	#*#*#*# TODO: Call your backtracking algorithm and time it #*#*#*#
	#================================================================#

    start_time = time.time()

    solved_board = backtracking(board)

    end_time = time.time()

    runtime = end_time - start_time

    return solved_board, runtime

    #=================================#
	#*#*#*# Your code ends here #*#*#*#
	#=================================#

def print_stats(runtimes):
    '''
    Prints a statistical summary of the runtimes of all the boards
    '''
    min = 100000000000
    max = 0
    sum = 0
    n = len(runtimes)

    for runtime in runtimes:
        sum += runtime
        if(runtime < min):
            min = runtime
        if(runtime > max):
            max = runtime

    mean = sum/n

    sum_diff_squared = 0

    for runtime in runtimes:
        sum_diff_squared += (runtime-mean)*(runtime-mean)

    std_dev = np.sqrt(sum_diff_squared/n)

    print("\nRuntime Statistics:")
    print("Number of Boards = {:d}".format(n))
    print("Min Runtime = {:.8f}".format(min))
    print("Max Runtime = {:.8f}".format(max))
    print("Mean Runtime = {:.8f}".format(mean))
    print("Standard Deviation of Runtime = {:.8f}".format(std_dev))
    print("Total Runtime = {:.8f}".format(sum))


if __name__ == '__main__':
    if len(sys.argv) > 1:

        # Running futoshiki solver with one board $python3 futoshiki.py <input_string>.
        print("\nInput String:")
        print(sys.argv[1])
        
        print("\nFormatted Input Board:")
        board = Board(sys.argv[1])
        board.print_board()
        
        solved_board, runtime = solve_board(board)
        
        print("\nSolved String:")
        print(solved_board.get_config_str())
        
        print("\nFormatted Solved Board:")
        solved_board.print_board()
        
        print_stats([runtime])

        # Write board to file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        outfile.write(solved_board.get_config_str())
        outfile.write('\n')
        outfile.close()

    else:
        # Running futoshiki solver for boards in futoshiki_start.txt $python3 futoshiki.py

        #  Read boards from source.
        src_filename = 'futoshiki_start.txt'
        try:
            srcfile = open(src_filename, "r")
            futoshiki_list = srcfile.read()
            srcfile.close()
        except:
            print("Error reading the sudoku file %s" % src_filename)
            exit()

        # Setup output file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        
        runtimes = []

        # Solve each board using backtracking
        for line in futoshiki_list.split("\n"):
            
            print("\nInput String:")
            print(line)
            
            print("\nFormatted Input Board:")
            board = Board(line)
            board.print_board()
            
            solved_board, runtime = solve_board(board)
            runtimes.append(runtime)
            
            print("\nSolved String:")
            print(solved_board.get_config_str())
            
            print("\nFormatted Solved Board:")
            solved_board.print_board()

            # Write board to file
            outfile.write(solved_board.get_config_str())
            outfile.write('\n')

        # Timing Runs
        print_stats(runtimes)
        
        outfile.close()
        print("\nFinished all boards in file.\n")
