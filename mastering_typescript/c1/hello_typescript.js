"use strict";
exports.__esModule = true;
var inquirer = require("inquirer");
inquirer.prompt([
    {
        name: "first_name",
        message: "What is your name?"
    }
]).then(function (answers) {
    console.log("You answered ".concat(answers.first_name));
});
