# Flopy optimization service

This is the repository for the flopy optimization service, that is used by INOWAS. It is build as a composition of 
docker containers that handle all the tasks between receiving an optimization request and pushing out the final result.

## Structure

The service consists of app, manager and worker(s). The app works as a connection between the optimization.inowas.com 
interface and the databases, the manager manages ongoing optimization tasks and the worker simply works. The following 
part describes the interactions more detailed.

### The app

The app can receive the following requests:
* optimization request: A request json that holds the modflow model data and further optimization information. The app 
evaluates the information and separates it into data written temporarly on the disk and an entry in the database waiting
to be finished. The entries are ordered specifically which means first a whole optimization task has to be finished, 
then the next one and so on. 

* stop request: A request to stop an optimization and return the current best solution. Simple request by calling an
url with the specific optimization id.

* progess request: A request to show the progress of the certain optimization that is presented to the user on the 
platform.

### The manager

The manager takes the given task and data to create individual jobs. As the optimization is a genetic one, the manager
has to creat slightly different tasks many times. The manager manages the creation of random parameters for each 
generation as well as the summary of finished generations. Thus it will only pick one optimization task and set it to
running and then follow the progress until each individual calculation has finished to sum up the generation and each
generation has finished to sum up the total optimization. As a last step it will put the optimization to be finished.

### The worker

The worker has one job only, which is to calculate created tasks. Therefor he scans the optimization tasks table to 
get the currently running optimization (if there's one) and then starts scanning for tasks for that optimization. It 
then grabs onem puts it to "under progress" and calculates it. The worker is the only part that has access to the 
modflow model as it's the only container where modflow is installed. It is also able to calculate the fitness of the 
task as kind of a response value.

