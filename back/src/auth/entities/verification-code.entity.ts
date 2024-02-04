import { Column, Entity, PrimaryColumn, Timestamp } from "typeorm";
import { v4 as uuidv4 } from 'uuid';

@Entity()
export class VerificationCode {

    constructor(email: string, code: string){
        this.VerificationCodeId = uuidv4();
        this.email = email;
        this.code = code;
    }

    @PrimaryColumn({type: 'uuid'})
    VerificationCodeId: string;

    @Column({ length: 50 })
    email: string;

    @Column()
    code: string;

    @Column({ type: 'timestamp with time zone', default: () => 'CURRENT_TIMESTAMP' })
    created_date: Timestamp;
}