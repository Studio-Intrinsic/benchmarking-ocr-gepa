#!/usr/bin/env node

/**
 * JSON Evaluation Script
 *
 * Calculates accuracy score for JSON extraction by comparing predicted JSON
 * to ground truth. Counts field-level differences (additions, deletions, modifications)
 * and returns a score between 0 and 1.
 *
 * This is the reference implementation used for all evaluation to ensure consistency.
 * Can be called from Python via subprocess.
 *
 * Usage:
 *   node evaluate_json.js '{"a":1}' '{"a":1}' false
 *   SIMPLE_OUTPUT=true node evaluate_json.js '{"a":1}' '{"a":2}' false
 */

const { diff } = require("json-diff");

/**
 * Calculates accuracy for JSON structure and primitive values only
 */
const calculateJsonAccuracy = (actual, predicted, ignoreCases = false) => {
  // Convert strings to uppercase if ignoreCases is true
  const processedActual = ignoreCases
    ? convertStringsToUppercase(actual)
    : actual;
  const processedPredicted = ignoreCases
    ? convertStringsToUppercase(predicted)
    : predicted;

  // Get the diff result
  const fullDiffResult = diff(processedActual, processedPredicted, {
    full: true,
    sort: true,
  });
  const diffResult = diff(processedActual, processedPredicted, { sort: true });
  const totalFields = countTotalFields(processedActual);

  if (!diffResult) {
    // If there's no diff, the JSONs are identical
    return {
      score: 1,
      jsonDiff: {},
      fullJsonDiff: {},
      jsonDiffStats: {
        additions: 0,
        deletions: 0,
        modifications: 0,
        total: 0,
      },
      totalFields,
    };
  }

  const changes = countChanges(diffResult);
  const score = Math.max(
    0,
    1 -
      (changes.additions + changes.deletions + changes.modifications) /
        totalFields
  );

  return {
    score: Number(score.toFixed(4)),
    jsonDiff: diffResult,
    fullJsonDiff: fullDiffResult,
    jsonDiffStats: changes,
    totalFields,
  };
};

/**
 * Recursively converts all string values in an object to uppercase
 */
const convertStringsToUppercase = (obj) => {
  if (obj === null || typeof obj !== "object") {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map((item) => convertStringsToUppercase(item));
  }

  const result = {};
  for (const key in obj) {
    const value = obj[key];
    if (typeof value === "string") {
      result[key] = value.toUpperCase();
    } else if (typeof value === "object" && value !== null) {
      result[key] = convertStringsToUppercase(value);
    } else {
      result[key] = value;
    }
  }
  return result;
};

/**
 * Count the number of field changes in the diff result
 * Returns counts for additions, deletions, modifications, and total
 */
const countChanges = (diffResult) => {
  const changes = {
    additions: 0,
    deletions: 0,
    modifications: 0,
    total: 0,
  };

  const traverse = (obj) => {
    if (!obj || typeof obj !== "object") {
      return;
    }

    for (const key in obj) {
      const value = obj[key];

      if (Array.isArray(value)) {
        // Handle array diffs
        value.forEach((item) => {
          // Check if item is in the expected [operation, element] format
          if (!Array.isArray(item) || item.length !== 2) {
            return;
          }

          const [operation, element] = item;
          if (element === null || typeof element !== "object") {
            // Handle primitive value changes in arrays
            switch (operation) {
              case "+":
                changes.additions++;
                break;
              case "-":
                changes.deletions++;
                break;
            }
          } else {
            switch (operation) {
              // Handle array element additions and deletions
              case "+":
                changes.additions += countTotalFields(element);
                break;
              case "-":
                changes.deletions += countTotalFields(element);
                break;
              case "~":
                // Handle array element modifications
                traverse(element);
                break;
            }
          }
        });
      } else {
        if (key.endsWith("__deleted")) {
          if (value === null || typeof value !== "object") {
            changes.deletions++;
          } else {
            changes.deletions += countTotalFields(value);
          }
        } else if (key.endsWith("__added")) {
          if (value === null || typeof value !== "object") {
            changes.additions++;
          } else {
            changes.additions += countTotalFields(value);
          }
        } else if (typeof value === "object" && value !== null) {
          if (value.__old !== undefined && value.__new !== undefined) {
            if (value.__old === null && value.__new !== null) {
              changes.modifications += countTotalFields(value.__new) || 1;
            } else {
              changes.modifications += countTotalFields(value.__old) || 1;
            }
          } else {
            traverse(value);
          }
        }
      }
    }
  };

  traverse(diffResult);

  changes.total = changes.additions + changes.deletions + changes.modifications;
  return changes;
};

/**
 * Count total number of primitive fields in a JSON object
 * Used to calculate the denominator for the accuracy score
 */
function countTotalFields(obj) {
  let count = 0;

  const traverse = (current) => {
    if (!current || typeof current !== "object") {
      return;
    }

    if (Array.isArray(current)) {
      // Traverse into array elements if they're objects
      current.forEach((item) => {
        if (typeof item === "object" && item !== null) {
          traverse(item);
        } else {
          count++;
        }
      });
    } else {
      for (const key in current) {
        // Skip diff metadata keys
        if (key.includes("__")) {
          continue;
        }

        // Only count primitive value fields
        if (
          current[key] === null ||
          typeof current[key] === "string" ||
          typeof current[key] === "number" ||
          typeof current[key] === "boolean"
        ) {
          count++;
        }
        // Recurse into nested objects and arrays
        else if (typeof current[key] === "object") {
          traverse(current[key]);
        }
      }
    }
  };

  traverse(obj);
  return count;
}

// CLI interface for calling from Python or command line
if (require.main === module) {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    console.error(
      "Usage: node evaluate_json.js <actual_json> <predicted_json> [ignore_case]"
    );
    process.exit(1);
  }

  try {
    const actual = JSON.parse(args[0]);
    const predicted = JSON.parse(args[1]);
    const ignoreCases = args[2] === "true";

    const result = calculateJsonAccuracy(actual, predicted, ignoreCases);

    // Output just the score for simple Python integration
    if (process.env.SIMPLE_OUTPUT === "true") {
      console.log(result.score);
    } else {
      // Output full result as JSON for detailed analysis
      console.log(JSON.stringify(result, null, 2));
    }
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
}

module.exports = {
  calculateJsonAccuracy,
  countTotalFields,
  countChanges,
  convertStringsToUppercase,
};
