🧭 Complete MongoDB Tutorial
============================

**Learn MongoDB and its ecosystem step-by-step --- from zero to production-ready expert.**

* * * * *

📘 MODULE 1: Database & NoSQL Fundamentals
------------------------------------------

### 🎯 Learning Goals

-   Understand databases, their purpose, and types.

-   Learn how MongoDB fits in the NoSQL world.

-   Grasp JSON, BSON, and MongoDB architecture basics.

* * * * *

### 1.1 What is a Database?

A **database** is an organized collection of data that can be easily accessed, managed, and updated.

#### Types of Databases

| Type | Example | Description |
| --- | --- | --- |
| Relational (SQL) | MySQL, PostgreSQL | Structured, tabular data with fixed schema. |
| Non-relational (NoSQL) | MongoDB, Cassandra | Flexible, schema-less data, optimized for scale. |

MongoDB belongs to the **Document-Oriented NoSQL** family.

* * * * *

### 1.2 MongoDB at a Glance

-   Stores data in **JSON-like documents** (BSON --- Binary JSON).

-   Schema can vary across documents.

-   Horizontal scaling using **sharding**.

-   High availability via **replica sets**.

* * * * *

### 1.3 Installing MongoDB

#### ✅ Option 1: macOS

`brew tap mongodb/brew
brew install mongodb-community@7.0
brew services start mongodb-community@7.0`

#### ✅ Option 2: Docker

`docker run -d -p 27017:27017 --name mongo mongo:7.0`

Verify installation:

`mongosh`

* * * * *

### 1.4 Basic Terminology

| SQL Term | MongoDB Term |
| --- | --- |
| Database | Database |
| Table | Collection |
| Row | Document |
| Column | Field |

* * * * *

### 🧩 Exercise 1

1.  Create a new database named `school`.

2.  Create a collection named `students`.

3.  Insert 3 sample student documents:

`db.students.insertMany([
  { name: "Alice", age: 20, major: "Physics" },
  { name: "Bob", age: 21, major: "Math" },
  { name: "Charlie", age: 22, major: "CS" }
])`

1.  Fetch all:

`db.students.find()`

* * * * *

📗 MODULE 2: CRUD OPERATIONS & QUERY MASTER
-------------------------------------------

### 🎯 Learning Goals

-   Perform Create, Read, Update, Delete operations.

-   Master query filters, projections, and indexes.

* * * * *

### 2.1 Create Operations

`db.students.insertOne({ name: "David", age: 23, major: "Biology" })
db.students.insertMany([
  { name: "Eve", age: 24, major: "Chemistry" },
  { name: "Frank", age: 25, major: "Math" }
])`

* * * * *

### 2.2 Read Operations

#### Basic Query:

`db.students.find({ major: "Math" })`

#### Projection:

`db.students.find({}, { name: 1, major: 1 })`

#### Comparison Operators:

`db.students.find({ age: { $gt: 21 } })`

#### Logical Operators:

`db.students.find({ $or: [{ major: "CS" }, { age: { $lt: 22 } }] })`

* * * * *

### 2.3 Update Operations

`db.students.updateOne(
  { name: "Alice" },
  { $set: { major: "Data Science" } }
)
db.students.updateMany(
  { major: "Math" },
  { $inc: { age: 1 } }
)`

* * * * *

### 2.4 Delete Operations

`db.students.deleteOne({ name: "Eve" })
db.students.deleteMany({ age: { $gt: 23 } })`

* * * * *

### 2.5 Indexes

`db.students.createIndex({ name: 1 })
db.students.getIndexes()`

Check performance:

`db.students.find({ name: "Alice" }).explain("executionStats")`

* * * * *

### 🧩 Exercise 2

-   Insert 10 student documents.

-   Query all students older than 21 sorted by age descending.

-   Create an index on `major`.

-   Use `.explain()` to measure query performance.

* * * * *

📙 MODULE 3: SCHEMA DESIGN & AGGREGATION
----------------------------------------

### 🎯 Learning Goals

-   Learn embedding vs referencing.

-   Design scalable schemas.

-   Use Aggregation Framework.

* * * * *

### 3.1 Embedding vs Referencing

#### Embedded (Denormalized)

`{
  name: "Alice",
  courses: [
    { title: "Math 101", grade: "A" },
    { title: "CS 101", grade: "B" }
  ]
}`

#### Referenced (Normalized)

`// students
{ _id: 1, name: "Alice" }
// courses
{ student_id: 1, title: "Math 101", grade: "A" }`

* * * * *

### 3.2 Schema Design Patterns

-   **Bucket Pattern**: Group time-series data.

-   **Subset Pattern**: Embed frequently accessed subset of data.

-   **Extended Reference Pattern**: Avoid excessive joins.

* * * * *

### 3.3 Aggregation Framework

Example:

`db.students.aggregate([
  { $match: { age: { $gte: 20 } } },
  { $group: { _id: "$major", avgAge: { $avg: "$age" } } },
  { $sort: { avgAge: -1 } }
])`

* * * * *

### 🧩 Exercise 3

-   Build a `posts` collection with user references.

-   Find the top 3 users with the most posts.

-   Use `$lookup` to join user data.

* * * * *

📕 MODULE 4: ADVANCED CONCEPTS
------------------------------

### 🎯 Learning Goals

-   Understand transactions, replication, and sharding.

-   Learn performance and scaling best practices.

* * * * *

### 4.1 Transactions

`session = db.getMongo().startSession()
session.startTransaction()
students = session.getDatabase("school").students
students.updateOne({ name: "Alice" }, { $inc: { age: 1 } })
session.commitTransaction()
session.endSession()`

* * * * *

### 4.2 Replication

-   Replica set = primary + secondaries.

-   Provides failover & read scaling.

Commands:

`mongod --replSet rs0
mongosh
rs.initiate()
rs.status()`

* * * * *

### 4.3 Sharding

-   Split data horizontally for scale.

`sh.enableSharding("school")
sh.shardCollection("school.students", { major: 1 })`

* * * * *

### 4.4 Backup & Restore

`mongodump --out /backup
mongorestore /backup`

* * * * *

📙 MODULE 5: MONGODB ATLAS & CLOUD
----------------------------------

### 🎯 Learning Goals

-   Use Atlas to deploy, monitor, and scale clusters.

-   Explore Data API, Triggers, and Vector Search.

* * * * *

### 5.1 Setup

-   Create free cluster on cloud.mongodb.com

-   Whitelist your IP.

-   Connect via VSCode or driver URI.

* * * * *

### 5.2 Data API Example

```curl -X POST\
  'https://data.mongodb-api.com/app/data-abcde/endpoint/data/v1/action/find'\
  -H 'Content-Type: application/json'\
  -H 'api-key: <YOUR_API_KEY>'\
  -d '{"dataSource":"Cluster0","database":"school","collection":"students"}'
```

* * * * *

### 5.3 Atlas Vector Search

For GenAI / RAG:

`db.students.createIndex({ embeddings: "vector" }, { vectorDimensions: 768, vectorType: "float32" })`

* * * * *

📗 MODULE 6: DEVELOPER INTEGRATIONS
-----------------------------------

### 🎯 Learning Goals

-   Connect MongoDB with backend frameworks.

* * * * *

### 6.1 Python Integration

```from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client.school
students = db.students.find()
for s in students:
    print(s)
```

* * * * *

### 6.2 Node.js Integration (Mongoose)

`npm install mongoose`

```import mongoose from "mongoose";
mongoose.connect("mongodb://localhost:27017/school");

const Student = mongoose.model("Student", { name: String, age: Number });
await Student.create({ name: "Zara", age: 22 });
```

* * * * *

### 🧩 Project 6

Build a **Task Manager API** with:

-   Express + Mongoose

-   CRUD endpoints

-   Connect to Atlas cluster

* * * * *

📕 MODULE 7: ECOSYSTEM, DEVOPS & ANALYTICS
------------------------------------------

### 🎯 Learning Goals

-   Deploy MongoDB apps, automate backups, visualize data.

* * * * *

### 7.1 MongoDB with Docker & Kubernetes

```# docker-compose.yml
services:
  mongo:
    image: mongo:7.0
    ports:
      - "27017:27017"
```

* * * * *

### 7.2 MongoDB Charts

-   Connect to Atlas cluster

-   Build dashboard for user analytics

* * * * *

### 7.3 GitHub Actions Backup Workflow

```name: Backup MongoDB
on:
  schedule:
    - cron: "0 3 * * *"
jobs:
  backup:
    runs-on: ubuntu-latest
    steps:
      - name: Dump DB
        run: mongodump --uri ${{ secrets.MONGO_URI }} --out ./backup
```
