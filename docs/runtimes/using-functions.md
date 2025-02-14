
# Creating and Using Functions

**MLRun Functions** (function objects) can be created by using any of the following methods:

- **{py:func}`~mlrun.run.new_function`**: creates a function from code repository/archive.
- **{py:func}`~mlrun.run.code_to_function`**: creates a function from local or remote source code (single file) or from 
 a notebook (code file will be embedded in the function object).
- **{py:func}`~mlrun.run.import_function`**: imports a function from a local or remote YAML function-configuration file or 
  from a function object in the MLRun database (using a DB address of the format `db://<project>/<name>[:<tag>]`)
  or from the function marketplace (e.g. `hub://describe`). See [MLRun Functions Marketplace](./load-from-marketplace.md).

When you create a function, you can:
- Use the {py:meth}`~mlrun.runtimes.BaseRuntime.save` function method to save a function object in the MLRun database.
- Use the {py:meth}`~mlrun.runtimes.BaseRuntime.export` method to save a YAML function-configuration to your preferred 
local or remote location.
- Use the {py:meth}`~mlrun.runtimes.BaseRuntime.run` to execute a task.
- Use the {py:meth}`~mlrun.runtimes.BaseRuntime.as_step` to convert a function to a Kubeflow pipeline step.
- Use `.deploy()` to build/deploy the function. (Deploy for batch functions builds the image and adds the required packages. 
For online/real-time runtimes like `nuclio` and `serving` it also deploys it as an online service.)

Functions are stored in the project and are versioned so you can always view previous code and go back to previous functions if needed.

The general concepts described in this section are illustrated in the following figure:

<img src="../_static/images/mlrun_function_flow.png" alt="functions-flow" width="800"/>

Read more in:
* [**Providing Function Code**](#providing-function-code)
* [**Specifying the function’s execution handler or command**](#specifying-the-function-execution-handler-or-command)
* [**Submitting Tasks/Jobs To Function**](#submitting-tasks/jobs-to-functions)
* [**MLRun Execution Context**](#mlrun-execution-context)
* [**Function Runtimes**](#function-runtimes)

## Providing Function Code

When using `code_to_function()` or `new_function()`, you can provide code in several ways:
- [As part of the function object](#provide-code-as-part-of-the-function-object)
- [As part of the function image](#provide-code-as-part-of-the-function-image)
- [From the git/zip/tar archive into the function at runtime](#provide-code-from-a-git-zip-tar-archive-into-the-function-at-runtime)

### Provide code as part of the function object
This method is great for small and single file functions or for using code derived from notebooks. This example uses the mlrun 
{py:func}`~mlrun.code_to_function` method to create functions from code files or notebooks. 
For more on how to create functions from notebook code, see [Converting notebook code to a function](./mlrun_code_annotations.ipynb).

    # create a function from py or notebook (ipynb) file, specify the default function handler
    my_func = mlrun.code_to_function(name='prep_data', filename='./prep_data.py', kind='job', 
    image='mlrun/mlrun', handler='my_func')

### Provide code as part of the function image

Providing code as part of the image is good for ensuring that the function image has the integrated code + dependencies, 
and it avoids the dependency, or overhead, of loading code at runtime. 

Use the {py:meth}`~mlrun.runtimes.KubejobRuntime.deploy()` method to build a function image with source code, 
dependencies, etc. Specify the build configuration using the {py:meth}`~mlrun.runtimes.KubejobRuntime.build_config` method. 

```
    # create a new job function from base image and archive + custom build commands
    fn = mlrun.new_function('archive', kind='job', command='./myfunc.py')
    fn.build_config(base_image='mlrun/mlrun', source='git://github.com/org/repo.git#master',
                    commands=["pip install pandas"])
    # deploy (build the container with the extra build commands/packages)
    fn.deploy()
    
    # run the function (specify the function handler to execute)
    run_results = fn.run(handler='my_func', params={"x": 100})
```

Alternatively, you can use a pre-built image:

```
# provide a pre-built image with your code and dependencies
fn = mlrun.new_function('archive', kind='job', command='./myfunc.py', image='some/pre-built-image:tag')
    
# run the function (specify the function handler to execute)
run_results = fn.run(handler='my_func', params={"x": 100})
```

You can use this option with {py:func}`~mlrun.run.new_function` method.


### Provide code from a git, zip, tar archive into the function at runtime

This option is the most efficient when doing iterative development with multiple code files and packages. You can 
make small code changes and re-run the job without building images, etc. You can use this option with the 
{py:func}`~mlrun.run.new_function` method.

The `local`, `job`, `mpijob` and `remote-spark` runtimes support dynamic load from archive or file shares (other 
runtimes will be added later). Enable this by setting the `spec.build.source=<archive>` and 
`spec.build.load_source_on_run=True` 
or simply by setting the `source` attribute in `new_function`). In the CLI, use the `--source` flag. 

    fn = mlrun.new_function('archive', kind='job', image='mlrun/mlrun', command='./myfunc.py', 
                            source='git://github.com/mlrun/ci-demo.git#master')
    run_results = fn.run(handler='my_func', params={"x": 100})

See more details and examples on [running jobs with code from Archives or shares](./code-archive.ipynb)

## Specifying the function execution handler or command

The function is configured with code and dependencies, however you also need to set the main execution code 
either by handler or command.

**Handler**

A handler is a method (not a script) that executes the function, for either a one-time run or ongoing online services.  

**Command**

The `command='./myfunc.py'` specifies the command that is executed in the function container/workdir. 

By default MLRun tries to execute python code with the specified command. For executing non-python code, set 
`mode="pass"` (passthrough) and specify the full execution `command`, e.g.:

    new_function(... command="bash main.sh --myarg xx", mode="pass")  
    
If you need to add arguments in the command, use `"mode=args"`  template (`{..}`) in the command to pass the 
task parameters as arguments for the execution command, for example:

    new_function(... command='mycode.py' --x {xparam}", mode="args")
    
where `{xparam}` is substituted with the value of the `xparam` parameter.<br>
It is possible to use argument templates also when using `mode="pass"`.

See also [Execute non Python code](./code-archive.html#execute-non-python-code) and 
[Inject parameters into command line](./code-archive.html#inject-parameters-into-command-line).


## Submitting Tasks/Jobs To Functions

MLRun batch function objects support a {py:meth}`~mlrun.runtimes.BaseRuntime.run` method for invoking a job over them. 
The run method accepts various parameters such as `name`, `handler`, `params`, `inputs`, `schedule`, etc. 
Alternatively you can pass a **`Task`** object (see: {py:func}`~mlrun.model.new_task`) that holds all of the 
parameters plus the advanced options. 

> **Run/simulate functions locally:** 
Functions can also run and be debugged locally by using the `local` runtime or by setting the `local=True` 
> parameter in the {py:meth}`~mlrun.runtimes.BaseRuntime.run` method (for batch functions).

Functions can host multiple methods (handlers). You can set the default handler per function. You
 need to specify which handler you intend to call in the run command. 
 
You can pass data objects to the function's `run()` method using the inputs dictionary argument, where the dictionary 
keys match the function's handler argument names and the MLRun data urls are provided as the values. The data is passed 
into the function as a {py:class}`~mlrun.datastore.DataItem` object that handles data movement, tracking and security in 
an optimal way. Read more about data objects in [Data Stores & Data Items](../store/datastore.md).

    run_results = fn.run(params={"label_column": "label"}, inputs={'data': data_url})

MLRun also supports iterative jobs that can run and track multiple child jobs (for hyper-parameter tasks, AutoML, etc.). 
See [Hyper-Param and Iterative jobs](../hyper-params.ipynb) for details and examples.
 
The `run()` command returns a run object that you can use to track the job and its results. If you
pass the parameter `watch=True` (default) the {py:meth}`~mlrun.runtimes.BaseRuntime.run` command blocks 
until the job completes.

Run object has the following methods/properties:
- `uid()` &mdash; returns the unique ID.
- `state()` &mdash; returns the last known state.
- `show()` &mdash; shows the latest job state and data in a visual widget (with hyperlinks and hints).
- `outputs` &mdash; returns a dictionary of the run results and artifact paths.
- `logs(watch=True)` &mdash; returns the latest logs.
    Use `Watch=False` to disable the interactive mode in running jobs.
- `artifact(key)` &mdash; returns an artifact for the provided key (as {py:class}`~mlrun.datastore.DataItem` object).
- `output(key)` &mdash; returns a specific result or an artifact path for the provided key.
- `wait_for_completion()` &mdash; wait for async run to complete
- `refresh()` &mdash; refresh run state from the db/service
- `to_dict()`, `to_yaml()`, `to_json()` &mdash; converts the run object to a dictionary, YAML, or JSON format (respectively).


<br>You can view the job details, logs. and artifacts in the user interface:

<br><img src="../_static/images/project-jobs-train-artifacts-test_set.png" alt="project-jobs-train-artifacts-test_set" width="800"/>


## MLRun Execution Context

In the function's handler code signature you can add context as the first argument. 
The context provides access to the job metadata, parameters, inputs, secrets, and API for logging and monitoring the results. 
Alternatively if it does not run inside a function handler (e.g. in Python main or Notebook) you can obtain the `context` 
object from the environment using the {py:func}`~mlrun.run.get_or_create_ctx` function.

Example function and usage of the context object:
 
```python
from mlrun.artifacts import ChartArtifact
import pandas as pd

def my_job(context, p1=1, p2="x"):
    # load MLRUN runtime context (will be set by the runtime framework)

    # get parameters from the runtime context (or use defaults)

    # access input metadata, values, files, and secrets (passwords)
    print(f"Run: {context.name} (uid={context.uid})")
    print(f"Params: p1={p1}, p2={p2}")
    print("accesskey = {}".format(context.get_secret("ACCESS_KEY")))
    print("file\n{}\n".format(context.get_input("infile.txt", "infile.txt").get()))

    # Run some useful code e.g. ML training, data prep, etc.

    # log scalar result values (job result metrics)
    context.log_result("accuracy", p1 * 2)
    context.log_result("loss", p1 * 3)
    context.set_label("framework", "sklearn")

    # log various types of artifacts (file, web page, table), will be versioned and visible in the UI
    context.log_artifact(
        "model",
        body=b"abc is 123",
        local_path="model.txt",
        labels={"framework": "xgboost"},
    )
    context.log_artifact(
        "html_result", body=b"<b> Some HTML <b>", local_path="result.html"
    )

    # create a chart output (will show in the pipelines UI)
    chart = ChartArtifact("chart")
    chart.labels = {"type": "roc"}
    chart.header = ["Epoch", "Accuracy", "Loss"]
    for i in range(1, 8):
        chart.add_row([i, i / 20 + 0.75, 0.30 - i / 20])
    context.log_artifact(chart)

    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
        "age": [42, 52, 36, 24, 73],
        "testScore": [25, 94, 57, 62, 70],
    }
    df = pd.DataFrame(raw_data, columns=["first_name", "last_name", "age", "testScore"])
    context.log_dataset("mydf", df=df, stats=True)
```

Example of creating the context objects from the environment:

```python
if __name__ == "__main__":
    context = mlrun.get_or_create_ctx('train')
    p1 = context.get_param('p1', 1)
    p2 = context.get_param('p2', 'a-string')
    # do something
    context.log_result("accuracy", p1 * 2)
    # commit the tracking results to the DB (and mark as completed)
    context.commit(completed=True)
```

Note that MLRun context is also a python context and can be used in a `with` statement (eliminating the need for `commit`)
d development
```python
if __name__ == "__main__":
    with mlrun.get_or_create_ctx('train') as context:
        p1 = context.get_param('p1', 1)
        p2 = context.get_param('p2', 'a-string')
        # do something
        context.log_result("accuracy", p1 * 2)
```

(Function_runtimes)=
## Function Runtimes

When you create an MLRun function you need to specify a runtime kind (e.g. `kind='job'`). Each runtime supports 
its own specific attributes (e.g. Jars for Spark, Triggers for Nuclio, Auto-scaling for Dask, etc.).

MLRun supports these runtimes:

Real-time runtimes:
* **nuclio** - real-time serverless functions over Nuclio
* **serving** - higher level real-time Graph (DAG) over one or more Nuclio functions

Batch runtimes:
* **handler** - execute python handler (used automatically in notebooks or for debug)
* **local** - execute a Python or shell program 
* **job** - run the code in a Kubernetes Pod
* **dask** - run the code as a Dask Distributed job (over Kubernetes)
* **mpijob** - run distributed jobs and Horovod over the MPI job operator, used mainly for deep learning jobs 
* **spark** - run the job as a Spark job (using Spark Kubernetes Operator)
* **remote-spark** - run the job on a remote Spark service/cluster (e.g. Iguazio Spark service)

**Common attributes for Kubernetes based functions** 

All the Kubernetes based runtimes (Job, Dask, Spark, Nuclio, MPIJob, Serving) support a common 
set of spec attributes and methods for setting the PODs:

function.spec attributes (similar to k8s pod spec attributes):
* volumes
* volume_mounts
* env
* resources
* replicas
* image_pull_policy
* service_account
* image_pull_secret

common function methods:
* set_env(name, value)
* set_envs(env_vars)
* gpus(gpus, gpu_type)
* with_limits(mem, cpu, gpus, gpu_type)
* with_requests(mem, cpu)
* set_env_from_secret(name, secret, secret_key)
