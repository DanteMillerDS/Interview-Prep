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

-- 2. JOIN Queries
-- Question 6: Retrieve employee names along with their department names using an INNER JOIN
SELECT e.employee_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;

-- Question 7: Use LEFT JOIN to retrieve all employees and their departments (if any)
SELECT e.employee_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;

-- 3. Aggregation Queries
-- Question 8: Calculate the total sales per product
SELECT product_id, SUM(sales) AS total_sales
FROM sales_data
GROUP BY product_id;

-- Question 9: Find the maximum salary in each department
SELECT department_id, MAX(salary) AS max_salary
FROM employees
GROUP BY department_id;

-- Question 10: Use HAVING to filter groups with total sales greater than 1000
SELECT product_id, SUM(sales) AS total_sales
FROM sales_data
GROUP BY product_id
HAVING total_sales > 1000;

-- 4. Window Functions
-- Question 11: Rank employees by salary within each department using ROW_NUMBER()
SELECT employee_name, department_id, salary,
ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS row_num
FROM employees;

-- Question 12: Calculate cumulative sales for each product ordered by date
SELECT product_id, order_date,
SUM(sales) OVER (PARTITION BY product_id ORDER BY order_date) AS cumulative_sales
FROM sales_data;

-- Question 13: Calculate a 7-day moving average of sales
SELECT order_date, 
AVG(sales) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS moving_avg
FROM sales_data;

-- 5. Advanced Window Functions
-- Question 14: Find the difference in sales between consecutive days using LAG()
SELECT product_id, order_date, sales,
sales - LAG(sales) OVER (PARTITION BY product_id ORDER BY order_date) AS sales_diff
FROM sales_data;

-- Question 15: Calculate the percentile rank of employees by salary within each department
SELECT employee_name, department_id, salary,
PERCENT_RANK() OVER (PARTITION BY department_id ORDER BY salary) AS percentile_rank
FROM employees;