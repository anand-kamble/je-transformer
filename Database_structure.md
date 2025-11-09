| table_name     | column_name                                   | data_type                | is_nullable | column_default |
| -------------- | --------------------------------------------- | ------------------------ | ----------- | -------------- |
| entry_line     | entry_line_id                                 | uuid                     | NO          |                |
| entry_line     | journal_entry_id                              | uuid                     | NO          |                |
| entry_line     | ledger_account_id                             | uuid                     | YES         |                |
| entry_line     | index                                         | integer                  | NO          |                |
| entry_line     | description                                   | text                     | NO          |                |
| entry_line     | debit                                         | numeric                  | NO          |                |
| entry_line     | credit                                        | numeric                  | NO          |                |
| entry_line     | exchange_rate                                 | numeric                  | NO          |                |
| journal_entry  | journal_entry_id                              | uuid                     | NO          |                |
| journal_entry  | business_id                                   | text                     | NO          |                |
| journal_entry  | year_month                                    | integer                  | NO          |                |
| journal_entry  | date                                          | timestamp with time zone | NO          |                |
| journal_entry  | number                                        | integer                  | NO          |                |
| journal_entry  | description                                   | text                     | NO          |                |
| journal_entry  | currency                                      | character                | NO          |                |
| journal_entry  | journal_entry_type                            | text                     | NO          |                |
| journal_entry  | journal_entry_sub_type                        | text                     | YES         |                |
| journal_entry  | journal_entry_status                          | USER-DEFINED             | NO          |                |
| journal_entry  | journal_entry_origin                          | USER-DEFINED             | NO          |                |
| ledger_account | ledger_account_id                             | uuid                     | NO          |                |
| ledger_account | business_id                                   | text                     | NO          |                |
| ledger_account | number                                        | text                     | NO          |                |
| ledger_account | name                                          | text                     | NO          |                |
| ledger_account | parent_ledger_account_id                      | uuid                     | YES         |                |
| ledger_account | currency                                      | character                | NO          |                |
| ledger_account | nature                                        | character                | NO          |                |
| ledger_account | ledger_account_type                           | text                     | YES         |                |
| ledger_account | ledger_account_sub_type                       | text                     | YES         |                |
| ledger_account | ledger_account_sub_sub_type                   | text                     | YES         |                |
| ledger_account | ledger_account_status                         | USER-DEFINED             | NO          |                |
| ledger_account | foreign_exchange_adjustment_ledger_account_id | uuid                     | YES         |                |
| ledger_account | cash_flow_group                               | USER-DEFINED             | YES         |                |
| ledger_account | added_date                                    | timestamp with time zone | NO          |                |
| ledger_account | removed_date                                  | timestamp with time zone | YES         |                |
| ledger_account | is_used_in_journal_entries                    | boolean                  | NO          |                |