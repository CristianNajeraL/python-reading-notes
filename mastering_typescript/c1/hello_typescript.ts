import * as inquirer from "inquirer";

inquirer.prompt(
    [
        {
            name: "first_name",
            message: "What is your name?"
        }
    ]
).then(
    answers => {
        console.log(`You answered ${answers.first_name}`);
    }
);