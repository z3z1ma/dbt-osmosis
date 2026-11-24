select account_id, status as account_status
from {{ ref('accounts') }}
