This project builds a binary file executing a python script which serves a summarization model on nvidia machine with GPU on PORT 5000
Follow the below steps to build the project.

1. Build the docker on an nvidia GPU equipped machine using the following command:
   `docker build -t build_emerse_app .`
2. Run the docker to build the binary file which executes the python script using the following command:
   `docker run -d --name build_emerse_app_container build_emerse_app`
3. Move the output created binary in dist from the docker into the application project.
   `docker cp build_emerse_app_container:app/dist ../test_binary/resources/`

If you changed the code and want to test it first before building the executable:

1. Uncomment the last line to run the python file, and comment run pyinstaller command in dockerfile
2. build the docker:
   `docker build -t build_emerse_app .`
3. Run the docker
`docker run -d --name build_emerse_app_container build_emerse_app --mount type=bind,source=../test_binary/,target=/app`
