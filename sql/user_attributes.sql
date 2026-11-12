SELECT
    user_id,
    any(os) AS os,
    any(source) AS source,
    any(gender) AS gender,
    any(age) AS age
FROM {db}.feed_actions
WHERE exp_group IN ({groups})
  AND toDate(time) BETWEEN '{start}' AND '{end}'
GROUP BY user_id
