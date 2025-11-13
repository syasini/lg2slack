# TODO Tracking System

This directory contains organized tracking for code fixes and enhancements.

## 📁 Files

- **`fixes.md`** - Critical bugs and code smells that need fixing
- **enhancements.md** - Refactoring opportunities and code elegance improvements

## 🎯 Quick Status

### Fixes (Critical)
- [ ] Issue #1: Extract duplicate reaction methods (HIGH)
- [ ] Issue #2: Move import re to module level (MEDIUM)
- [ ] Issue #3: Switch to loguru logging (MEDIUM)

### Enhancements (Nice-to-Have)
- [ ] Opportunity #1: Refactor _normalize_reactions() (HIGH)
- [ ] Opportunity #2: Simplify buffer flush logic (MEDIUM)
- [ ] Opportunity #3: Add configuration validation (MEDIUM)

## 🔄 Workflow

1. Pick an item from `fixes.md` or `enhancements.md`
2. Create a new branch (branch name listed in the file)
3. Make the changes
4. Run tests: `pytest -x`
5. Commit and push
6. Update the markdown file status
7. Create PR

## 📊 Priority Order

**Do First:**
1. Issue #1 (Duplicate reaction methods) - Biggest impact
2. Issue #2 (Import re) - Quick win
3. Opportunity #1 (Refactor normalize) - Code clarity

**Do Later:**
4. Issue #3 (Loguru) - Nice to have
5. Opportunity #2 (Buffer flush) - Minor improvement
6. Opportunity #3 (Validation) - Safety improvement

## 🎨 Code Quality Principles

- **DRY** (Don't Repeat Yourself) - Eliminate duplication
- **Single Responsibility** - Each function does one thing
- **PEP 8** - Follow Python style guide
- **Type Hints** - Use typing annotations
- **Test Coverage** - Every change should have tests

## 📝 Notes

- Each fix/enhancement should be on its own branch
- Branch naming: `claude/<description>-011CUomKh1oKXEihWTyV7Sra`
- Always run full test suite before committing
- Update status in markdown files as you go
