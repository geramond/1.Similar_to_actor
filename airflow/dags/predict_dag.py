from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum


default_args = {
    "owner": "geramond",
    # "start_date": days_ago(1),  # запуск день назад
    "start_date": pendulum.today('UTC').add(days=-1),  # запуск день назад
    "retries": 5,  # запуск таска до 5 раз, если ошибка
    #"retry_delay": datetime.timedelta(minutes=5),  # дельта запуска при повторе 5 минут
    "task_concurency": 1  # одновременно только 1 таск
}

params = {
    'name': 'predict',
    'name_dag': 'predict_dag',
    'schedule': '2 * * * *'
}

def init_dag(dag, task_id):
    with dag:
        t1 = BashOperator(
            task_id=f"{task_id}",
            bash_command=f"python3 /Users/maksimfomin/IT/DS_practice/4.CV/1.Similar_to_actor/{params['name']}.py")
    return dag


dag = DAG(params['name_dag'],
          schedule=params['schedule'],
          max_active_runs=1,
          default_args=default_args
          )

init_dag(dag, params['name_dag'])
globals()[params['name_dag']] = dag
