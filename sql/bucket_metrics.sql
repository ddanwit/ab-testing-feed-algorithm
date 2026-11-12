SELECT
    exp_group,
    bucket,
    sum(likes) / sum(views) AS bucket_ctr,
    median(ctr) AS ctr_median
FROM (
    SELECT
        exp_group,
        xxHash64(user_id) % {n_buckets} AS bucket,
        user_id,
        sum(action = 'like') AS likes,
        sum(action = 'view') AS views,
        likes / views AS ctr
    FROM {db}.feed_actions
    WHERE toDate(time) BETWEEN '{start}' AND '{end}'
      AND exp_group IN ({groups})
    GROUP BY exp_group, bucket, user_id
)
GROUP BY exp_group, bucket
