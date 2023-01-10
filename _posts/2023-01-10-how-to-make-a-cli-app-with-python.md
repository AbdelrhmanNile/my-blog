---
title: How to make a CLI app with Python?
date: 2023-01-09 21:31:00 +02:00
categories: [tutorial]
tags: [python, cli]
---

Hi! Welcome to my first blog post. In this post, I will show you how to make a CLI app with Python.

First, let's talk about what a CLI app is and why you should use it.

## What is a CLI app?
CLI stands for Command Line Interface. A CLI app is an app that runs in the terminal. For example, the `ls` command is a CLI app that lists the files in the current directory.

## Why use a CLI app?
There are many reasons to use a CLI app. For example, you can use it to automate a task that you do often.

well enough, let's get started!

## How to make a CLI app with Python?
There are a lot of ways to make a CLI app with Python. In this tutorial we will do it the easy way. We will use the `typer` library.

### Installing typer
To install typer, run the following command:
```console
$ pip install typer[all]
```

### Creating the app
Now, let's create the app. Create a new file called `app.py`:
```console
$ touch app.py
```
we are going to create a simple CLI app that prints "Hello World!".

Open the `app.py` file and follow me:
```python
# first we import the typer library!
import typer

# then we create an instance from the typer.Typer class, let's call it cli.
cli = typer.Typer()

# now we create a function that will be called when we run the app.
# we will use the @cli.command() decorator to tell typer that this function is a command.
@cli.command()
def hello():
    print("Hello World!")

# finally, we run the app.
if __name__ == "__main__":
    cli()
```
{: file='app.py'}
now let's run the app:
```console
$ python app.py
Hello World!
```

well that's it! you have made your first CLI app with Python. Now you can add more commands to your app. Because we only had one command, `hello`, typer made it the default command. So, if you run the app without any arguments, it will run the `hello` command.
Let's add another command to our app that prints my name:

```python
# first we import the typer library!
import typer

# then we create an instance from the typer.Typer class, let's call it cli.
cli = typer.Typer()

# now we create a function that will be called when we run the app.
# we will use the @cli.command() decorator to tell typer that this function is a command.

# first command, named hello
@cli.command()
def hello():
    print("Hello World!")

# second command, named name
@cli.command()
def name():
    print("Abdelrhman Nile")

# finally, we run the app.
if __name__ == "__main__":
    cli()
```
{: file='app.py'}

now let's run the app, and see what happens:
```console
$ python app.py
Usage: app.py [OPTIONS] COMMAND [ARGS]...
Try "app.py --help" for help.
```
oh no! we got an error! now we need to specify which command we want to run. Let's run the `name` command:
```console
$ python app.py name
Abdelrhman Nile
```
hmm good, but boring!!!! we can do better. Let's try to make the name a variable:
```python
# first we import the typer library!
import typer

# then we create an instance from the typer.Typer class, let's call it cli.
cli = typer.Typer()

# now we create a function that will be called when we run the app.
# we will use the @cli.command() decorator to tell typer that this function is a command.

# first command, named hello
@cli.command()
def hello():
    print("Hello World!")

# second command, named name
# takes a your_name argument
@cli.command()
def name(your_name: str):
    print(f"Hello {your_name}!")

# finally, we run the app.
if __name__ == "__main__":
    cli()
```
{: file='app.py'}

now let's run the app, and see what happens:
```console
$ python app.py name
Usage: app.py name [OPTIONS] YOUR_NAME
Try "app.py name --help" for help.
Error: Missing argument "YOUR_NAME".
```
oh no! we got an error again! now we need to specify the name. Let's run the `name` command:
```console
$ python app.py name Abdelrhman
Abdelrhman
```

## Conclusion
In this tutorial, we learned how to make a CLI app with Python. We used the `typer` library to make it easy. We also learned how to add arguments to our commands.

This should get you started with making CLI apps with Python, but there is a lot more to learn! you can make much more advanced CLI apps with `typer`. I might write another tutorial about it soon; until then, you can read the `typer` [documentation](https://typer.tiangolo.com/).

Thanks for reading! I hope you enjoyed this tutorial. If you have any questions, feel free to ask them in the comments. I will try to answer them as soon as possible.