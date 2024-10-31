-- SQL Practice Questions

-- 1. Basic SELECT Queries
-- Question 1: Retrieve all records from a table
SELECT * FROM employees;

-- Question 2: Retrieve specific columns (employee_name and salary) from the employees table
SELECT employee_name, salary FROM employees;

-- Question 3: Retrieve unique values from a column
SELECT DISTINCT department_id FROM employees;

-- Question 4: Filter records using a WHERE clause
SELECT employee_name, salary 
FROM employees 
WHERE salary > 50000;

-- Question 5: Use ORDER BY to sort the result set by salary in descending order
SELECT employee_name, salary 
FROM employees 
ORDER BY salary DESC;

-- Question 6: Limit the number of rows returned
SELECT employee_name, salary 
FROM employees 
ORDER BY salary DESC 
LIMIT 5;

-- 2. Handling NULL Values and Conditional Expressions
-- Question 7: Use COALESCE to replace NULL values with a default value
SELECT employee_name, COALESCE(bonus, 0) AS bonus 
FROM employees;

-- Question 8: Use NULLIF to return NULL if two values are equal
SELECT employee_name, 
       NULLIF(bonus, 0) AS bonus_if_not_zero 
FROM employees;

-- Question 9: Use CASE to categorize employees based on salary
SELECT employee_name, 
       CASE 
           WHEN salary > 80000 THEN 'High'
           WHEN salary BETWEEN 50000 AND 80000 THEN 'Medium'
           ELSE 'Low'
       END AS salary_category 
FROM employees;

-- 3. Aggregation Queries
-- Question 10: Calculate the total salary of all employees
SELECT SUM(salary) AS total_salary FROM employees;

-- Question 11: Find the average salary of employees
SELECT AVG(salary) AS avg_salary FROM employees;

-- Question 12: Count the total number of employees
SELECT COUNT(*) AS employee_count FROM employees;

-- Question 13: Group employees by department and calculate total salary per department
SELECT department_id, SUM(salary) AS total_salary 
FROM employees 
GROUP BY department_id;

-- Question 14: Use HAVING to filter groups with total salary greater than 100000
SELECT department_id, SUM(salary) AS total_salary 
FROM employees 
GROUP BY department_id 
HAVING total_salary > 100000;

-- 4. Join Queries
-- Question 15: Retrieve employee names along with their department names using an INNER JOIN
SELECT e.employee_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;

-- Question 16: Use LEFT JOIN to retrieve all employees and their departments (if any)
SELECT e.employee_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;

-- Question 17: Use RIGHT JOIN to retrieve all departments and their employees (if any)
SELECT e.employee_name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.department_id;

-- Question 18: Use FULL OUTER JOIN to retrieve all employees and all departments
SELECT e.employee_name, d.department_name
FROM employees e
FULL OUTER JOIN departments d ON e.department_id = d.department_id;

-- 5. Subqueries and Common Table Expressions (CTEs)
-- Question 19: Use a subquery to find employees with above-average salaries
SELECT employee_name, salary 
FROM employees 
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Question 20: Use a CTE to list departments with more than 5 employees
WITH DepartmentCounts AS (
    SELECT department_id, COUNT(*) AS employee_count
    FROM employees
    GROUP BY department_id
)
SELECT d.department_name, dc.employee_count
FROM departments d
JOIN DepartmentCounts dc ON d.department_id = dc.department_id
WHERE dc.employee_count > 5;

-- 6. Window Functions
-- Question 21: Rank employees by salary within each department using ROW_NUMBER()
SELECT employee_name, department_id, salary,
ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank
FROM employees;

-- Question 22: Calculate the cumulative sum of salaries for each department
SELECT employee_name, department_id, salary,
SUM(salary) OVER (PARTITION BY department_id ORDER BY employee_name) AS cumulative_salary
FROM employees;

-- Question 23: Use LAG to find the previous employee’s salary within the same department
SELECT employee_name, department_id, salary,
LAG(salary) OVER (PARTITION BY department_id ORDER BY employee_name) AS previous_salary
FROM employees;

-- Question 24: Use LEAD to find the next employee’s salary within the same department
SELECT employee_name, department_id, salary,
LEAD(salary) OVER (PARTITION BY department_id ORDER BY employee_name) AS next_salary
FROM employees;

-- 7. Set Operations
-- Question 25: Use UNION to combine two result sets
SELECT employee_name FROM employees WHERE department_id = 1
UNION
SELECT employee_name FROM employees WHERE job_role = 'Manager';

-- Question 26: Use INTERSECT to find employees who work in both departments 1 and 2
SELECT employee_name FROM employees WHERE department_id = 1
INTERSECT
SELECT employee_name FROM employees WHERE department_id = 2;

-- Question 27: Use EXCEPT to find employees who are not managers
SELECT employee_name FROM employees
EXCEPT
SELECT employee_name FROM employees WHERE job_role = 'Manager';

-- 8. Advanced Query Techniques
-- Question 28: Use a correlated subquery to find employees whose salary is greater than the average salary of their department
SELECT employee_name, salary 
FROM employees e
WHERE salary > (
    SELECT AVG(salary) 
    FROM employees 
    WHERE department_id = e.department_id
);

-- Question 29: Use a window function to calculate the percentile rank of employees by salary
SELECT employee_name, salary,
PERCENT_RANK() OVER (ORDER BY salary) AS percentile_rank
FROM employees;

-- Question 30: Use COALESCE to handle NULL values in a salary bonus report
SELECT employee_name, COALESCE(bonus, 0) AS bonus
FROM employees;

-- Question 31: Calculate a 7-day rolling average of 'value' in a time series
SELECT 
    date,
    value,
    AVG(value) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_avg_7_days
FROM 
    observations;

-- Question 32: Handle NULL values in numeric and categorical columns by filling with mean and mode
UPDATE data
SET num_column = (SELECT AVG(num_column) FROM data)
WHERE num_column IS NULL;

UPDATE data
SET cat_column = (
    SELECT cat_column 
    FROM data 
    GROUP BY cat_column 
    ORDER BY COUNT(*) DESC 
    LIMIT 1
)
WHERE cat_column IS NULL;

-- Question 33: Calculate average time difference between consecutive events for each user
WITH OrderedEvents AS (
    SELECT 
        user_id,
        event_time,
        LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_event_time
    FROM 
        events
),
TimeDifferences AS (
    SELECT 
        user_id,
        TIMESTAMPDIFF(SECOND, prev_event_time, event_time) AS time_diff
    FROM 
        OrderedEvents
    WHERE 
        prev_event_time IS NOT NULL
)
SELECT 
    user_id, 
    AVG(time_diff) AS avg_time_diff
FROM 
    TimeDifferences
GROUP BY 
    user_id;
