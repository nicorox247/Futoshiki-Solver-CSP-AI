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

        queue = list(reassigned_variables)
        while queue:
            variable = queue.pop(0)
            value = self.config[variable]

            row, col = ROW.index(variable[0]), COL.index(variable[1])

            # Apply inequality constraints
            inequalities = self.get_inequality_neighbors(variable)
            for neighbor, inequality in inequalities:
                domain_before = self.domains[neighbor][:]
                if not self.apply_inequality_forward_check(variable, neighbor, inequality):
                    return False
                if self.domains[neighbor] != domain_before and neighbor not in queue and self.config[neighbor] == 0:
                    queue.append(neighbor)

            if value != 0:
                # Remove the value from the domains of variables in the same row
                for j in range(self.n):
                    neighbor = ROW[row] + COL[j]
                    if neighbor != variable and value in self.domains[neighbor]:
                        self.domains[neighbor].remove(value)
                        if not self.domains[neighbor]:
                            return False  # Domain wiped out
                        if neighbor not in queue and self.config[neighbor] == 0:
                            queue.append(neighbor)

                # Remove the value from the domains of variables in the same column
                for i in range(self.n):
                    neighbor = ROW[i] + COL[col]
                    if neighbor != variable and value in self.domains[neighbor]:
                        self.domains[neighbor].remove(value)
                        if not self.domains[neighbor]:
                            return False
                        if neighbor not in queue and self.config[neighbor] == 0:
                            queue.append(neighbor)

        return True

    def apply_inequality_forward_check(self, variable1, variable2, inequality):
        domain1 = self.domains[variable1]
        domain2 = self.domains[variable2]

        if inequality == '<':
            new_domain1 = [val1 for val1 in domain1 if any(val1 < val2 for val2 in domain2)]
            new_domain2 = [val2 for val2 in domain2 if any(val2 > val1 for val1 in new_domain1)]
        elif inequality == '>':
            new_domain1 = [val1 for val1 in domain1 if any(val1 > val2 for val2 in domain2)]
            new_domain2 = [val2 for val2 in domain2 if any(val2 < val1 for val1 in new_domain1)]
        else:
            return True  # No inequality to enforce

        if not new_domain1 or not new_domain2:
            return False  # Domain wipeout

        # Update domains if they have changed
        if new_domain1 != domain1:
            self.domains[variable1] = new_domain1
        if new_domain2 != domain2:
            self.domains[variable2] = new_domain2

        return True
        
    def get_inequality_neighbors(self, variable):
        '''
        Returns a list of tuples (neighbor_variable, inequality) for all inequality constraints involving the given variable.
        The inequality is from variable to neighbor.
        '''
        row, col = ROW.index(variable[0]), COL.index(variable[1])
        inequalities = []

        # Check horizontal inequalities
        # Inequality to the right
        if col < self.n - 1:
            inequality_key = variable + '*'
            if inequality_key in self.config and self.config[inequality_key] in ['<', '>']:
                neighbor = ROW[row] + COL[col + 1]
                inequality = self.config[inequality_key]
                inequalities.append((neighbor, inequality))
        # Inequality from the left
        if col > 0:
            inequality_key = ROW[row] + COL[col - 1] + '*'
            if inequality_key in self.config and self.config[inequality_key] in ['<', '>']:
                neighbor = ROW[row] + COL[col - 1]
                inequality = self.config[inequality_key]
                # Reverse the inequality since it's from neighbor to variable
                reverse_inequality = '<' if inequality == '>' else '>'
                inequalities.append((neighbor, reverse_inequality))

        # Check vertical inequalities
        # Inequality below
        if row < self.n - 1:
            inequality_key = variable[0] + '*' + variable[1]
            if inequality_key in self.config and self.config[inequality_key] in ['<', '>']:
                neighbor = ROW[row + 1] + COL[col]
                inequality = self.config[inequality_key]
                inequalities.append((neighbor, inequality))
        # Inequality from above
        if row > 0:
            inequality_key = ROW[row - 1] + '*' + COL[col]
            if inequality_key in self.config and self.config[inequality_key] in ['<', '>']:
                neighbor = ROW[row - 1] + COL[col]
                inequality = self.config[inequality_key]
                # Reverse the inequality since it's from neighbor to variable
                reverse_inequality = '<' if inequality == '>' else '>'
                inequalities.append((neighbor, reverse_inequality))

        return inequalities
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
    
    def update_config_str(self):
        '''
        Updates the configuration string based on the current board configuration.
        '''
        config_string = ''
        for i in range(0, self.n):
            for j in range(0, self.n):
                config_string += str(self.config[ROW[i] + COL[j]])
                if j != self.n - 1:
                    config_string += self.config[ROW[i] + COL[j] + '*']
            if i != self.n - 1:
                for j in range(0, self.n):
                    config_string += self.config[ROW[i] + '*' + COL[j]]
        self.config_str = config_string

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
    unassigned_vars = [var for var in board.get_variables() if board.config[var] == 0]
    if not unassigned_vars:
        if board.row_constraint() and board.column_constraint() and board.binary_constraint():
            return board
        else:
            return None

    mrv_var = min(unassigned_vars, key=lambda var: len(board.domains[var]))

    # Copy the domain to avoid modifying the board's domains
    original_domain = board.domains[mrv_var][:]

    for value in original_domain:
        # Backup the current state
        domains_backup = copy.deepcopy(board.domains)
        config_backup = board.config.copy()

        # Assign the value to the variable
        board.config[mrv_var] = value

        # Check if the assignment is consistent with constraints
        if not board.row_constraint() or not board.column_constraint() or not board.binary_constraint():
            # Restore state and continue
            board.config = config_backup
            board.domains = domains_backup
            continue

        # Perform forward checking after the assignment
        if not board.forward_checking([mrv_var]):
            # Forward checking failed, restore state and continue
            board.config = config_backup
            board.domains = domains_backup
            continue

        # Recursively call backtracking
        result = backtracking(board)
        if result:
            return result

        # Backtrack: undo the assignment and restore domains
        board.config = config_backup
        board.domains = domains_backup

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

    if solved_board:
        solved_board.update_config_str()

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
