create sequence if not exists storage_project_id_seq as bigint
  minvalue 1000000 maxvalue 4294967294 start with 1000000 no cycle;

create table if not exists storage_accounts (
  owner_user_id text primary key references "user"(id) on delete restrict,
  project_id bigint not null unique default nextval('storage_project_id_seq')
    check (project_id between 1 and 4294967294),
  limit_bytes bigint not null default 21474836480 check (limit_bytes > 0),
  state text not null default 'pending' check (state in ('pending', 'ready', 'error')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists storage_reservations (
  id uuid primary key default gen_random_uuid(),
  owner_user_id text not null references storage_accounts(owner_user_id) on delete restrict,
  operation_key text not null check (length(operation_key) between 1 and 128),
  kind text not null check (kind in ('upload', 'import', 'label_upload', 'label_publish',
    'train', 'evaluate', 'predict', 'batch_predict', 'index', 'transfer')),
  reserved_bytes bigint not null check (reserved_bytes > 0),
  created_at timestamptz not null default now(),
  released_at timestamptz,
  unique (owner_user_id, operation_key)
);

create index if not exists storage_reservations_active_owner
  on storage_reservations(owner_user_id) where released_at is null;

-- This is an accounted upper bound on the kernel limit, not the user's plan quota.
alter table storage_accounts add column if not exists enforced_limit_bytes bigint
  not null default 65536 check (enforced_limit_bytes >= 1024 and enforced_limit_bytes % 1024 = 0);
alter table storage_accounts add column if not exists breakdown jsonb not null default '{}';
alter table storage_accounts add column if not exists reconciled_at timestamptz;

create table if not exists storage_transfers (
  id uuid primary key,
  source_user_id text not null references "user"(id) on delete cascade,
  target_user_id text not null references "user"(id) on delete cascade,
  lease_id text not null check (lease_id ~ '^[a-f0-9]{32}$'),
  state text not null check (state in ('copying','committed','completed','rolled_back')),
  plan jsonb not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (source_user_id <> target_user_id)
);
create index if not exists storage_transfers_pending on storage_transfers(state)
  where state in ('copying','committed');
