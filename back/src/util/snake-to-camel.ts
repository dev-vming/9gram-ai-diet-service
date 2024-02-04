export function snakeTocamel(obj:Object) {
    const keys = Object.keys(obj);
    const values = Object.values(obj);
    const newObj = {};
    for(let i=0; i<keys.length; i++){
        const key = keys[i].split('_').map((word, index) => {
            if(index === 0) return word;
            return word[0].toUpperCase() + word.slice(1);
        }).join('');
        newObj[key] = values[i];
    }
    return newObj;
}