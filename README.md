# LIPIKAR OCR for Indian Languages

Brief description of your project.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Instructions on how to install and set up your project.

### Option 1: Install `maskrcnn-benchmark`

#### Prerequisites

Before you can use this project, you need to install `maskrcnn-benchmark`. You can do this by following the instructions in the [official documentation](https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/INSTALL.md). 

1. Clone the repository:

git clone https://github.com/your/repository

2. Navigate to the repository:

cd repository

3. Install the required Python packages:

pip install -r requirements.txt


### Option 2: Install Docker

If you prefer to use Docker, follow these steps:

1. Install Docker by following the instructions in the [official documentation](https://docs.docker.com/get-docker/).

git clone https://github.com/your/repository

2. Navigate to the repository:

cd repository

* Download trained model (contourNet_model.pth) from [here](https://csciitd-my.sharepoint.com/:f:/g/personal/ch7190150_iitd_ac_in/EvLT451TYFpAmJglzzsISJYBzvlnbMeVn4lhnJg07xM4Qw?e=PbYDTJ)
* Run the following commands in the terminal
    * ```docker build -t ocr_docker .```
    * ```docker run -it --name ocr_app -v /path/to/inference:cn_infenerce (eg /home/asrar/Downloads/cn_inference:/cn_inference)  ocr_docker```
    * ```cd cn_inference```
    * ```export INSTALL_DIR=$PWD```
    * ```cd $INSTALL_DIR && cd maskrcnn-benchmark && python3 setup.py build develop && cd $INSTALL_DIR```
    * ```unset INSTALL_DIR```
    * ```cp -r maskrcnn-benchmark/maskrcnn_benchmark .```


Once you've completed either Option 1 or Option 2, you'll be ready to use this project.

## Usage

To start the app, run the following command in your terminal:

python3 run.py

This will start the app and you can access it in your web browser
You will be having two folders in the root directory of the project:
- `auth_app`: contains your Flask app
- `instance`: contains your database
## Project Directory Structure
The directory structure for this project is as follows:
```bash 
auth_app/
├── auth/
│ ├── forms.py
│ ├── routes.py
│ └── templates/
│ ├── login.html
│ ├── register.html
│ ├── reset_password.html
│ ├── reset_request.html
│ └── base.html
│ └── utils.py
├── static/
│ ├── bootstrap/
│ ├── js/
│ ├── iitd-logo.png
│ ├── profile_picture/
│ ├── uploads/
│ └── style.css
├── user/
│ ├── forms.py
│ ├── models.py
│ ├── routes.py
│ └── templates/
│ ├── account.html
│ ├── admin.html
│ ├── error.html
│ ├── choosefile.html
│ ├── history.html
│ ├── navbar.html
│ ├── upload.html
│ └── verify_users.html
│ └── utils.py
├── run.py
├── utils1.py
└── data.db
```

The `auth` folder contains all files related to user authentication, while the `user` folder contains files related to user management with authentication. 

Both `auth/routes.py` and `user/routes.py` contain functions that are called in the main Flask application. `forms.py` files in both folders contain functions for creating forms for user input. The `models.py` file in the `user` folder contains the structure of the database.

Please note that the `templates` directory contains HTML templates for rendering pages in the web application, while the `static` directory contains static files such as images, stylesheets, and JavaScript files.

## Database Management

To manage the database for this project, you will need to have SQLite3 installed on your system. Once you have SQLite3 installed, you can open the database in sqlite3

From here, you can perform various SQL commands to modify the data stored in the database.

Please note that there will always be one user designated as the super admin, with an ID of 0. When you add the first user to the system, you will need to manually update their ID to 0 in order to designate them as the super admin.

If you are not familiar with SQLite3 or SQL commands in general, we recommend reviewing relevant documentation and resources to ensure proper management of your database.

## Contributing

Guidelines on how to contribute to your project. Include information on coding standards, how to set up a development environment, and how to submit pull requests.

## License

Include information on the license for your project. For example:

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

