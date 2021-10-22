<ol>
<h4><li><a href="">
Q. Write SQL query to find the nth highest salary from a table without using the TOP/limit keyword.
</a></li></h4>

1. Using Top keyword(SQL Server)
```
SELECT TOP 1 Salary
FROM (
      SELECT DISTINCT TOP N Salary
      FROM Employee
      ORDER BY Salary DESC
      )
ORDER BY Salary ASC;

```

2. Using limit clause(MySQL)
```
SELECT Salary
FROM Employee
ORDER BY Salary DESC LIMIT N-1,1;
```

3. Without using the TOP/limit keyword
```
SELECT Salary
FROM EmployeeSalary Emp1
WHERE N-1 = (
                SELECT COUNT( DISTINCT ( Emp2.Salary ) )
                FROM EmployeeSalary Emp2
                WHERE Emp2.Salary > Emp1.Salary
            )
```


<h4><li><a href="">
Q. Write an SQL query to fetch top n records?
</a></li></h4>

```
SELECT *
FROM EmployeeSalary
ORDER BY Salary DESC LIMIT N;
```
```
SELECT TOP N *
FROM EmployeeSalary
ORDER BY Salary DESC;
```


<h4><li><a href="">
Q. Write an SQL query to fetch only odd rows from the table.
</a></li></h4>

```
SELECT * FROM EmployeeDetails
WHERE MOD (EmpId, 2) <> 0;
```

```
SELECT E.EmpId, E.Project, E.Salary
FROM (
    SELECT *, Row_Number() OVER(ORDER BY EmpId) AS RowNumber
    FROM EmployeeSalary
) E
WHERE E.RowNumber % 2 = 1;
```

<h4><li><a href="">
Q. Write an SQL query to remove duplicates from a table without using a temporary table.
</a></li></h4>

```
SELECT * FROM EmployeeDetails
WHERE MOD (EmpId, 2) = 0;
```
```
SELECT E.EmpId, E.Project, E.Salary
FROM (
    SELECT *, Row_Number() OVER(ORDER BY EmpId) AS RowNumber
    FROM EmployeeSalary
) E
WHERE E.RowNumber % 2 = 0;
```

<h4><li><a href="">
Q. Write an SQL query to remove duplicates from a table without using a temporary table.
</a></li></h4>

```
DELETE E1 FROM EmployeeDetails E1
INNER JOIN EmployeeDetails E2
WHERE E1.EmpId > E2.EmpId
AND E1.FullName = E2.FullName
AND E1.ManagerId = E2.ManagerId
AND E1.DateOfJoining = E2.DateOfJoining
AND E1.City = E2.City;
```
</ol>